"""Lightweight Temporal Transformer World Model for Network Attack Forecasting.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Implements a timestep-level Temporal Transformer that simultaneously:
  1. Forecasts continuous network state dynamics P(S_{t+1} | S_t) via the Transition Head.
  2. Predicts future multi-step threat risk probabilities via the Threat Forecasting Head across horizon k=5.
  3. Classifies MITRE ATT&CK progression stages (0..5) via the Neural MITRE Head.
  4. Exposes multi-head self-attention tensors for explainability and dynamic saliency.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
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

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs pre-LN attention and feed-forward, returning output and raw attention weights."""
        src_norm = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        src = src + self.dropout1(attn_out)

        src_norm = self.norm2(src)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_out)

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

    def get_pos_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns positional embedding adaptively matching the input sequence length."""
        if seq_len == self.pos_embedding.shape[1]:
            return self.pos_embedding.to(device)
        elif seq_len < self.pos_embedding.shape[1]:
            return self.pos_embedding[:, :seq_len, :].to(device)
        else:
            pos = self.pos_embedding.transpose(1, 2)
            pos_interp = nn.functional.interpolate(pos, size=seq_len, mode="linear", align_corners=False)
            return pos_interp.transpose(1, 2).to(device)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Temporal Transformer World Model.

        Args:
            x: Input continuous state tensor of shape (Batch, seq_len, num_features)

        Returns:
            Tuple of:
              - pred_s_next: Reconstructed next state tensor of shape (Batch, seq_len, num_features)
              - primary_attack_logit: Primary attack classification logit of shape (Batch, 1)
              - latent_embedding: Pooled latent representation of shape (Batch, 64)
              - mitre_logits: Neural MITRE ATT&CK stage logits of shape (Batch, num_mitre_classes)
              - attn_weights: Self-attention tensor
        """
        # Linear projection + learnable temporal positional embeddings
        x_clamped = torch.clamp(x, -5.0, 5.0)
        pos = self.get_pos_embedding(x.shape[1], x.device)
        h = self.project_input(x_clamped) + pos
        h = self.pos_dropout(h)

        # Transformer temporal encoding with attention extraction
        last_attn = None
        for layer in self.layers:
            h, last_attn = layer(h)

        # 1. State transition dynamics prediction S_{t+1}
        pred_s_next = self.transition_head(h)

        # 2. Temporal pooling across sequence length -> (Batch, 64)
        latent_embedding = h.mean(dim=1)

        # 3. Future risk horizon logits
        future_risk_logits = self.forecasting_head(latent_embedding)

        # 4. Primary attack logit for immediate classification -> (Batch, 1)
        primary_attack_logit = future_risk_logits[:, 0:1]

        # 5. Neural MITRE ATT&CK classification logits
        mitre_logits = self.mitre_head(latent_embedding)

        return pred_s_next, primary_attack_logit, latent_embedding, mitre_logits, last_attn

    @torch.no_grad()
    def predict_timeline(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts calibrated multi-step future threat probabilities across horizon k."""
        x_clamped = torch.clamp(x, -5.0, 5.0)
        pos = self.get_pos_embedding(x.shape[1], x.device)
        h = self.project_input(x_clamped) + pos
        for layer in self.layers:
            h, _ = layer(h)
        latent = h.mean(dim=1)
        risk_logits = self.forecasting_head(latent)
        probs = torch.sigmoid(risk_logits)

        # Low-threat dynamic horizon calibration:
        p0 = probs[:, 0:1]
        is_low_threat = (p0 < 0.30)
        if torch.any(is_low_threat):
            k = probs.size(1)
            decay_factors = torch.linspace(1.0, 0.80, steps=k, device=x.device, dtype=probs.dtype).unsqueeze(0)
            calibrated = torch.minimum(probs, p0 * decay_factors + 0.02)
            probs = torch.where(is_low_threat, calibrated, probs)

        return torch.clamp(probs, 0.0, 1.0)

    @torch.no_grad()
    def predict_mitre_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts MITRE ATT&CK stage probabilities across classes."""
        pos = self.get_pos_embedding(x.shape[1], x.device)
        h = self.project_input(x) + pos
        for layer in self.layers:
            h, _ = layer(h)
        latent = h.mean(dim=1)
        logits = self.mitre_head(latent)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def imagine(
        self,
        x: Optional[torch.Tensor] = None,
        horizon: int = 5,
        n_trajectories: int = 4,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Autoregressively rolls forward state predictions and forecasts timeline risk without observations."""
        device = next(self.parameters()).device
        if x is None:
            x = torch.zeros((1, self.seq_len, self.num_features), device=device)
        elif x.dim() == 2:
            x = x.unsqueeze(0).to(device)
        else:
            x = x.to(device)

        # Base predictions on observed context
        tl_tensor = self.predict_timeline(x).squeeze(0)
        tl = tl_tensor.cpu().numpy()
        mitre_p = self.predict_mitre_stage(x).squeeze(0).cpu().numpy()
        pred_stage = int(np.argmax(mitre_p))

        feature_forecast = []
        infilt_mean = []
        predicted_stages = []
        cur_x = x.clone()

        for step in range(horizon):
            pred_s, primary_logit, latent, mitre_l, _ = self.forward(cur_x)
            nxt_feat = pred_s[:, -1:, :]
            feature_forecast.append(nxt_feat.squeeze().cpu().numpy())

            if step < (len(tl) if tl.ndim > 0 else 1):
                step_risk = float(tl[step] if tl.ndim > 0 else tl)
            else:
                step_risk = float(torch.sigmoid(primary_logit).squeeze().cpu().item())
            infilt_mean.append(step_risk)

            m_stage = int(torch.argmax(torch.softmax(mitre_l, dim=-1), dim=-1).squeeze().cpu().item())
            predicted_stages.append(m_stage)

            if cur_x.shape[-1] != nxt_feat.shape[-1]:
                if not hasattr(self, "_expand_proj") or self._expand_proj.out_features != cur_x.shape[-1]:
                    self._expand_proj = nn.Linear(self.num_features, cur_x.shape[-1]).to(cur_x.device)
                nxt_feat_expanded = self._expand_proj(nxt_feat)
                cur_x = torch.cat([cur_x[:, 1:, :], nxt_feat_expanded], dim=1)
            else:
                cur_x = torch.cat([cur_x[:, 1:, :], nxt_feat], dim=1)

        # Low-threat dynamic horizon anchoring
        if len(infilt_mean) > 0 and infilt_mean[0] < 0.30:
            p0 = infilt_mean[0]
            infilt_mean = [min(infilt_mean[i], p0 * (1.0 - 0.02 * i) + 0.01) for i in range(len(infilt_mean))]

        infilt_lower = [max(0.0, infilt_mean[k] - (0.04 * (k + 1))) for k in range(horizon)]
        infilt_upper = [min(1.0, infilt_mean[k] + (0.04 * (k + 1))) for k in range(horizon)]

        return {
            "forecast_timeline": infilt_mean,
            "infilt_prob_mean": infilt_mean,
            "infilt_prob_lower": infilt_lower,
            "infilt_prob_upper": infilt_upper,
            "feature_forecast": feature_forecast,
            "predicted_stages": predicted_stages,
        }
