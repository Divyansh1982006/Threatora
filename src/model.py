"""Lightweight Temporal Transformer World Model for Network Attack Forecasting.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Implements a timestep-level Temporal Transformer that simultaneously:
  1. Forecasts continuous network state dynamics P(S_{t+1} | S_t) via the Transition Head.
  2. Predicts future multi-step threat risk probabilities via the Threat Forecasting Head.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class ThreatoraTemporalTransformerWorldModel(nn.Module):
    """Timestep-level Temporal Transformer World Model for continuous network flow telemetry."""

    def __init__(
        self,
        seq_len: int = 20,
        num_features: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon_k: int = 5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.horizon_k = horizon_k

        # 1. Input Projection & Learnable Positional Embeddings
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # 2. Temporal Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Transition Head: World Model State Dynamics P(S_{t+1} | S_t)
        # Maps encoded temporal representations (Batch, 20, 64) -> (Batch, 20, 12)
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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Temporal Transformer World Model.

        Args:
            x: Input continuous state tensor of shape (Batch, seq_len=20, num_features=12)

        Returns:
            Tuple of:
              - pred_s_next: Reconstructed next state tensor of shape (Batch, 20, 12)
              - primary_attack_logit: Primary attack classification logit of shape (Batch, 1)
              - latent_embedding: Pooled latent representation of shape (Batch, 64)
        """
        # Linear projection + learnable temporal positional embeddings
        h = self.input_proj(x) + self.pos_embedding
        h = self.pos_dropout(h)

        # Transformer temporal encoding: (Batch, 20, 64)
        encoded = self.transformer(h)

        # 1. State transition dynamics prediction S_{t+1}
        pred_s_next = self.transition_head(encoded)  # (Batch, 20, 12)

        # 2. Temporal pooling across sequence length -> (Batch, 64)
        latent_embedding = encoded.mean(dim=1)

        # 3. Future risk horizon logits -> (Batch, 5)
        future_risk_logits = self.forecasting_head(latent_embedding)

        # 4. Primary attack logit for immediate classification -> (Batch, 1)
        primary_attack_logit = future_risk_logits[:, 0:1]

        return pred_s_next, primary_attack_logit, latent_embedding

    @torch.no_grad()
    def predict_timeline(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts calibrated multi-step future threat probabilities across horizon k.

        Args:
            x: Continuous state tensor of shape (Batch, seq_len=20, num_features=12)

        Returns:
            Sigmoid threat probabilities of shape (Batch, horizon_k=5)
        """
        h = self.input_proj(x) + self.pos_embedding
        encoded = self.transformer(h)
        latent = encoded.mean(dim=1)
        risk_logits = self.forecasting_head(latent)
        return torch.sigmoid(risk_logits)
