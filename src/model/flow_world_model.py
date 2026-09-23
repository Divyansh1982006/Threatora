"""Flow-level LSTM World Model for Network Attack Forecasting and K-Step Rollout.

Implements:
1. Input normalization with LayerNorm(12)
2. 2-layer LSTM temporal encoder (hidden_dim=128, dropout=0.30)
3. Latent representation projection (latent_dim=64)
4. Multi-task heads:
   - Attack classification head -> P_flow(t)
   - Next-state transition dynamics head -> S_hat_{t+1} (12 features)
5. Autonomous K-step Monte Carlo forward simulation via .imagine()
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FLOW_FEATURE_NAMES: List[str] = [
    "flow_duration",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "ack_flag_number",
    "Tot size",
    "IAT",
    "Number",
]


class FlowLSTMWorldModel(nn.Module):
    """LSTM-based Deterministic World Model for 12-feature Flow Telemetry."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden: int = 128,
        layers: int = 2,
        latent: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = layers
        self.latent = latent
        self.dropout = dropout

        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(hidden, latent),
            nn.LayerNorm(latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attack_head = nn.Sequential(
            nn.Linear(latent, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.next_state_head = nn.Sequential(
            nn.Linear(latent, 64),
            nn.GELU(),
            nn.Linear(64, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass over a sequence of flow observations.

        Args:
            x: Tensor of shape (B, T, input_dim)
            hx: Optional hidden and cell state tuple ((layers, B, hidden), (layers, B, hidden))

        Returns:
            Dictionary with attack_logits (B,), attack_prob (B,), next_state (B, input_dim),
            latent (B, latent), and (h_n, c_n).
        """
        x_norm = self.input_norm(x)
        seq, (h_n, c_n) = self.encoder(x_norm, hx)
        last_hidden = seq[:, -1, :]
        z = self.to_latent(last_hidden)

        logits = self.attack_head(z).squeeze(-1)
        next_state = self.next_state_head(z)
        prob = torch.sigmoid(logits)

        return {
            "attack_logits": logits,
            "attack_prob": prob,
            "next_state": next_state,
            "latent": z,
            "hidden_state": h_n,
            "cell_state": c_n,
        }

    def forward_tuple(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible tuple interface matching training scripts."""
        out = self.forward(x)
        return out["attack_logits"], out["next_state"]

    def imagine(
        self,
        current_sequence: torch.Tensor,
        horizon: int = 10,
        n_trajectories: int = 16,
        mc_dropout: bool = True,
    ) -> Dict[str, Any]:
        """Performs K-step forward simulation rollout WITHOUT external observations.

        Autoregressively projects future flow states and computes attack probability
        trajectories with Monte Carlo uncertainty confidence intervals.

        Args:
            current_sequence: (1, T, 12) or (B, T, 12) recent observed sequence
            horizon: Number of forward lookahead steps K
            n_trajectories: Number of MC trajectories for variance estimation
            mc_dropout: If True, enables dropout during inference for epistemic uncertainty

        Returns:
            Dictionary containing forecast timeline, mean attack risk, lower/upper CI bounds,
            and forecasted 12-feature states.
        """
        was_training = self.training
        if mc_dropout:
            self.train()  # Enable dropout layers for MC spread
        else:
            self.eval()

        device = current_sequence.device
        if current_sequence.ndim == 2:
            current_sequence = current_sequence.unsqueeze(0)

        b, t, d = current_sequence.shape
        # Repeat for Monte Carlo trajectories: (B * n_trajectories, T, d)
        rep_seq = current_sequence.repeat_interleave(n_trajectories, dim=0)

        # First pass to establish initial recurrent state from history
        with torch.no_grad():
            out = self.forward(rep_seq)
            cur_next_state = out["next_state"]  # (B * n_traj, d)
            cur_h = out["hidden_state"]          # (layers, B * n_traj, hidden)
            cur_c = out["cell_state"]            # (layers, B * n_traj, hidden)

        all_probs = []
        all_forecast_states = []

        # Current sliding window state in rollout
        rolling_state = cur_next_state.unsqueeze(1)  # (B * n_traj, 1, d)

        for step in range(horizon):
            with torch.no_grad():
                step_out = self.forward(rolling_state, (cur_h, cur_c))
                prob_k = step_out["attack_prob"]     # (B * n_traj,)
                next_s = step_out["next_state"]     # (B * n_traj, d)
                cur_h = step_out["hidden_state"]
                cur_c = step_out["cell_state"]

            all_probs.append(prob_k)
            all_forecast_states.append(next_s)
            rolling_state = next_s.unsqueeze(1)

        if not was_training:
            self.eval()

        # Stack over horizon: (horizon, B * n_traj)
        probs_stack = torch.stack(all_probs, dim=0).cpu().numpy()
        states_stack = torch.stack(all_forecast_states, dim=0).cpu().numpy()

        # Mean and standard deviation across trajectories
        prob_mean = np.mean(probs_stack, axis=1)  # (horizon,)
        prob_std = np.std(probs_stack, axis=1)
        prob_upper = np.clip(prob_mean + 1.96 * prob_std, 0.0, 1.0)
        prob_lower = np.clip(prob_mean - 1.96 * prob_std, 0.0, 1.0)

        mean_feature_forecast = np.mean(states_stack, axis=1)  # (horizon, 12)

        timeline = []
        for k in range(horizon):
            p_mean = float(prob_mean[k])
            p_lo = float(prob_lower[k])
            p_hi = float(prob_upper[k])
            timeline.append({
                "step": k + 1,
                "lookahead": f"+{k+1} step",
                "timestamp": f"t+{k+1}m",
                "attack_prob": round(p_mean, 4),
                "infilt_prob": round(p_mean, 4),
                "predicted_risk_score": round(p_mean, 4),
                "lower_ci": round(p_lo, 4),
                "upper_ci": round(p_hi, 4),
                "lower_ci_95": round(p_lo, 4),
                "upper_ci_95": round(p_hi, 4),
                "is_predicted_attack": bool(p_mean >= 0.5),
                "projected_stage": "Impact / DoS" if p_mean >= 0.5 else "Benign Traffic"
            })

        return {
            "horizon": horizon,
            "prob_mean": prob_mean.tolist(),
            "prob_std": prob_std.tolist(),
            "prob_lower": prob_lower.tolist(),
            "prob_upper": prob_upper.tolist(),
            "feature_forecast": mean_feature_forecast.tolist(),
            "forecast_timeline": timeline,
        }
