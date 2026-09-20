"""Lightweight Temporal Transformer World Model for Network Attack Forecasting.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Implements a timestep-level Temporal Transformer that simultaneously:
  1. Forecasts continuous network state dynamics P(S_{t+1} | S_t) via the Transition Head.
  2. Predicts future multi-step threat risk probabilities via the Threat Forecasting Head across horizon k=5.
  3. Classifies MITRE ATT&CK progression stages (0..5) via the Neural MITRE Head.
  4. Exposes multi-head self-attention tensors for explainability and dynamic saliency.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class TemporalTransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with explicit multi-head self-attention tensor extraction."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass extracting self-attention weights of shape (B, nhead, S, S)."""
        src2, attn_weights = self.self_attn(
            src, src, src, need_weights=True, average_attn_weights=False
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class ThreatoraTemporalTransformerWorldModel(nn.Module):
    """Timestep-level Temporal Transformer World Model for continuous network flow telemetry."""

    def __init__(
        self,
        seq_len: int = 20,
        num_features: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon_k: int = 5,
        num_mitre_classes: int = 6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.horizon_k = horizon_k
        self.num_mitre_classes = num_mitre_classes

        # 1. Input Projection & Learnable Positional Embeddings
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # 2. Temporal Transformer Backbone with Attention Extraction
        self.layers = nn.ModuleList([
            TemporalTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 3. Transition Head: World Model State Dynamics P(S_{t+1} | S_t)
        # Maps encoded temporal representations (Batch, 20, 64) -> (Batch, 20, num_features)
        self.transition_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_features),
        )

        # 4. Threat Forecasting Head: Multi-step future risk horizon (horizon_k=5)
        self.forecasting_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, horizon_k),
        )

        # 5. Neural MITRE ATT&CK Classification Head (6 stages)
        # (0: Benign, 1: Reconnaissance, 2: Initial Access, 3: Lateral Movement, 4: C2, 5: Exfiltration)
        self.mitre_head = nn.Linear(d_model, num_mitre_classes)

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input features into d_model, adaptively handling dynamic feature dimensions."""
        in_dim = x.shape[-1]
        if in_dim == self.num_features:
            return self.input_proj(x)
        if not hasattr(self, "_adaptive_proj") or self._adaptive_proj.in_features != in_dim:
            self._adaptive_proj = nn.Linear(in_dim, self.d_model).to(x.device)
        return self._adaptive_proj(x)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Temporal Transformer World Model.

        Args:
            x: Input continuous state tensor of shape (Batch, seq_len=20, num_features)

        Returns:
            Tuple of:
              - pred_s_next: Reconstructed next state tensor of shape (Batch, 20, num_features)
              - primary_attack_logit: Primary attack classification logit of shape (Batch, 1)
              - latent_embedding: Pooled latent representation of shape (Batch, 64)
              - mitre_logits: Neural MITRE ATT&CK stage logits of shape (Batch, 6)
              - attn_weights: Self-attention tensor of shape (Batch, nhead, 20, 20)
        """
        # Linear projection + learnable temporal positional embeddings
        h = self.project_input(x) + self.pos_embedding
        h = self.pos_dropout(h)

        # Transformer temporal encoding with attention extraction
        last_attn = None
        for layer in self.layers:
            h, last_attn = layer(h)

        # 1. State transition dynamics prediction S_{t+1}
        pred_s_next = self.transition_head(h)  # (Batch, 20, num_features)

        # 2. Temporal pooling across sequence length -> (Batch, 64)
        latent_embedding = h.mean(dim=1)

        # 3. Future risk horizon logits -> (Batch, 5)
        future_risk_logits = self.forecasting_head(latent_embedding)

        # 4. Primary attack logit for immediate classification -> (Batch, 1)
        primary_attack_logit = future_risk_logits[:, 0:1]

        # 5. Neural MITRE ATT&CK classification logits -> (Batch, 6)
        mitre_logits = self.mitre_head(latent_embedding)

        return pred_s_next, primary_attack_logit, latent_embedding, mitre_logits, last_attn

    @torch.no_grad()
    def predict_timeline(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts calibrated multi-step future threat probabilities across horizon k.

        Args:
            x: Continuous state tensor of shape (Batch, seq_len=20, num_features)

        Returns:
            Sigmoid threat probabilities of shape (Batch, horizon_k=5)
        """
        h = self.project_input(x) + self.pos_embedding
        for layer in self.layers:
            h, _ = layer(h)
        latent = h.mean(dim=1)
        risk_logits = self.forecasting_head(latent)
        return torch.sigmoid(risk_logits)

    @torch.no_grad()
    def predict_mitre_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts MITRE ATT&CK stage probabilities across 6 classes."""
        h = self.project_input(x) + self.pos_embedding
        for layer in self.layers:
            h, _ = layer(h)
        latent = h.mean(dim=1)
        logits = self.mitre_head(latent)
        return torch.softmax(logits, dim=-1)
