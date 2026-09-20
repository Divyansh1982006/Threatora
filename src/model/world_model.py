"""LSTM-based Deterministic World Model for Network Attack Forecasting.

Learns network traffic state transition dynamics P(S_t+1 | S_t) over 60-second windows.
Unlike static classifiers, the model simulates future states without observations via .imagine().

Architecture:
    obs x_t --[Encoder]--> e_t
                             |
    (h_t, c_t) = LSTM( e_t, c_t-1 )     <-- Long Short-Term Memory dynamics
                             |
                  S_t = h_t
                    /    |    \\
            Decoder   Heads   Causal Attention (attention weights for explainability)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, default_model_config

def _build_mlp(layer_sizes: List[int], dropout: float = 0.0, final_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        is_last = i == len(layer_sizes) - 2
        if not is_last or final_norm:
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class CausalTemporalAttention(nn.Module):
    """Causal Multi-Head Attention over past hidden states.
    Upper triangle is masked so each step can only attend to current and past states.
    Produces attention weights used directly by the Explainability Engine.
    """

    def __init__(self, state_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (B, T, state_dim)
        Returns:
            context: (B, T, state_dim)
            weights: (B, T, T) attention weights averaged across heads
        """
        b, t, d = states.shape
        # Mask future steps: True values are ignored in PyTorch MultiheadAttention
        future_mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=states.device), diagonal=1)
        x_norm = self.norm(states)
        context, weights = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=future_mask,
            need_weights=True,
            average_attn_weights=True
        )
        return context, weights


class LSTMRecurrentCell(nn.Module):
    """LSTM transition dynamics: maintains hidden state h_t and cell state c_t."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.ln_h = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x_in: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = self.lstm_cell(x_in, hx)
        return self.ln_h(h), self.ln_c(c)


import warnings


class NetworkWorldModel(nn.Module):
    """[DEPRECATED] Legacy LSTM-based Model for Network Attack Forecasting.
    
    Deprecated in favor of ThreatoraTemporalTransformerWorldModel.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        warnings.warn(
            "NetworkWorldModel is deprecated and scheduled for removal. "
            "Use ThreatoraTemporalTransformerWorldModel from src.model.transformer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.cfg = cfg or default_model_config
        o, e, h = self.cfg.obs_dim, self.cfg.embed_dim, self.cfg.hidden_dim

        # 1. Observation Encoder: x_t (62) -> e_t (64)
        self.encoder = _build_mlp([o, e, e], dropout=self.cfg.dropout, final_norm=True)

        # 2. Recurrent Transition Dynamics: (h_t, c_t)
        self.rnn_cell = LSTMRecurrentCell(input_size=e, hidden_size=h)

        # State representation dimension: S_t = h_t
        self.state_dim = h

        # 3. Causal Temporal Attention over S_1..S_T
        self.attention = CausalTemporalAttention(self.state_dim, n_heads=self.cfg.n_heads, dropout=self.cfg.dropout)

        # 4. Prediction Heads (operate directly on latent state S_t)
        # Head A: Observation Reconstruction / Decoder: S_t -> \hat{x}_t
        self.decoder = _build_mlp([self.state_dim, e, o], dropout=0.0)

        # Head B: Infiltration Probability: S_t -> [0, 1]
        self.infilt_head = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.SiLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Head C: MITRE ATT&CK Stage Classifier: S_t -> 5 stages (Recon -> Exfil)
        self.stage_head = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.SiLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(64, self.cfg.num_stages + 1)  # 0=Benign, 1..5=Stages
        )

    def forward(
        self,
        observations: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Runs the LSTM over an observed sequence.

        Args:
            observations: (B, T, obs_dim) normalized 60s window matrices.
            initial_state: Optional tuple (h_0, c_0).
        """
        b, t, d = observations.shape
        device = observations.device
        h = self.cfg.hidden_dim

        # Initial states
        if initial_state is None:
            prev_h = torch.zeros(b, h, device=device)
            prev_c = torch.zeros(b, h, device=device)
        else:
            prev_h, prev_c = initial_state

        # Encode observations
        e_seq = self.encoder(observations.view(b * t, d)).view(b, t, -1)

        states_list = []

        # Step through time sequence
        for step in range(t):
            e_t = e_seq[:, step, :]
            next_h, next_c = self.rnn_cell(e_t, (prev_h, prev_c))

            state_t = next_h
            states_list.append(state_t)

            prev_h, prev_c = next_h, next_c

        states_tensor = torch.stack(states_list, dim=1)  # (B, T, state_dim)

        # Causal Attention over states
        attn_context, attn_weights = self.attention(states_tensor)
        combined_state = states_tensor + attn_context

        # Multi-task heads
        recon_x = self.decoder(combined_state)
        infilt_prob = self.infilt_head(combined_state).squeeze(-1)  # (B, T)
        stage_logits = self.stage_head(combined_state)              # (B, T, num_stages+1)

        return {
            "reconstruction": recon_x,
            "infiltration_prob": infilt_prob,
            "stage_logits": stage_logits,
            "states": states_tensor,
            "attention_weights": attn_weights,
            "final_lstm_h": prev_h,
            "final_lstm_c": prev_c
        }

    def imagine(
        self,
        initial_lstm_h: torch.Tensor,
        initial_lstm_c: torch.Tensor,
        horizon: int = 10,
        n_trajectories: int = 16,
        mc_dropout: bool = True
    ) -> Dict[str, Any]:
        """Performs K-step forward simulation from current state WITHOUT observations.

        Uses MC-dropout to generate stochastic spread across trajectories.
        Runs heads on dreamed states to produce risk forecast & uncertainty bands.
        """
        was_training = self.training
        if mc_dropout:
            self.train()  # Enable dropout layers for MC-Dropout

        device = initial_lstm_h.device

        # Expand for Monte Carlo trajectories
        cur_h = initial_lstm_h.repeat_interleave(n_trajectories, dim=0)
        cur_c = initial_lstm_c.repeat_interleave(n_trajectories, dim=0)

        imagined_states = []
        all_infilt_probs = []
        all_stage_preds = []
        all_feature_forecasts = []

        # Start with an initial zero-observation encoding, or predict from h_0?
        # A simple zero token is effective for autonomous rollouts.
        e_dim = self.cfg.embed_dim
        cur_e = torch.zeros(cur_h.size(0), e_dim, device=device)

        for step in range(horizon):
            # Recurrent step
            cur_h, cur_c = self.rnn_cell(cur_e, (cur_h, cur_c))

            state_k = cur_h
            imagined_states.append(state_k)

            # Predictions on imagined state
            prob_k = self.infilt_head(state_k).squeeze(-1)         # (B * n_traj,)
            stage_logits_k = self.stage_head(state_k)              # (B * n_traj, num_stages+1)
            stage_k = torch.argmax(stage_logits_k, dim=-1)
            recon_k = self.decoder(state_k)                        # (B * n_traj, obs_dim)

            all_infilt_probs.append(prob_k)
            all_stage_preds.append(stage_k)
            all_feature_forecasts.append(recon_k)

            # To sustain autoregressive flow, we use the reconstructed observation as the next input
            cur_e = self.encoder(recon_k)

        if mc_dropout and not was_training:
            self.eval()

        # Stack over horizon steps: shape (horizon, B * n_traj) -> reshape to (horizon, n_traj)
        infilt_stack = torch.stack(all_infilt_probs, dim=0).cpu().detach().numpy()
        stages_stack = torch.stack(all_stage_preds, dim=0).cpu().detach().numpy()
        forecast_feats_stack = torch.stack(all_feature_forecasts, dim=0).cpu().detach().numpy()

        # Mean and std across Monte Carlo trajectories
        prob_mean = np.mean(infilt_stack, axis=1)
        prob_std = np.std(infilt_stack, axis=1)
        prob_upper = np.clip(prob_mean + 1.96 * prob_std, 0.0, 1.0)
        prob_lower = np.clip(prob_mean - 1.96 * prob_std, 0.0, 1.0)

        # Majority vote stage per horizon step
        stage_trajectory = []
        for step in range(horizon):
            counts = np.bincount(stages_stack[step], minlength=self.cfg.num_stages + 1)
            majority_stage = int(np.argmax(counts))
            stage_trajectory.append(majority_stage)

        mean_feature_forecast = np.mean(forecast_feats_stack, axis=1)  # (horizon, obs_dim)

        return {
            "horizon": horizon,
            "infilt_prob_mean": prob_mean.tolist(),
            "infilt_prob_std": prob_std.tolist(),
            "infilt_prob_upper": prob_upper.tolist(),
            "infilt_prob_lower": prob_lower.tolist(),
            "predicted_stages": stage_trajectory,
            "feature_forecast": mean_feature_forecast.tolist()
        }
