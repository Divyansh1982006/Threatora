"""Packet-Level LSTM World Model for Network Attack Forecasting and K-Step Rollout.

Architecture:
1. 20-Feature Network State Input (5-second aggregated packet telemetry)
2. 2-layer LSTM temporal encoder (hidden_dim=128, dropout=0.30)
3. Shared latent representation projection (latent_dim=64)
4. Multi-task heads:
   - Risk Head -> P_packet(t) Infiltration probability [0, 1]
   - Future State Head -> 5-step forecasted 20-feature packet states (5, 20)
   - Stage Head -> 6-class MITRE ATT&CK stage logits
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PACKET_FEATURE_NAMES: List[str] = [
    "packet_count",
    "byte_count",
    "flow_count",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_ports",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "ttl_mean",
    "ttl_std",
    "payload_mean",
    "payload_std",
    "window_mean",
    "window_std",
    "iat_mean",
    "iat_std",
]

PACKET_MITRE_STAGES: List[str] = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
]


class PacketLSTMWorldModel(nn.Module):
    """LSTM-based Deterministic World Model for 20-feature Packet Telemetry."""

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        prediction_horizon: int = 5,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prediction_horizon = prediction_horizon

        # ---------------- LSTM Encoder ----------------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ---------------- Shared Latent Space ----------------
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ---------------- Future State Prediction Head ----------------
        self.future_head = nn.Linear(64, prediction_horizon * input_size)

        # ---------------- Binary Risk Prediction Head (Sigmoid) ----------------
        self.risk_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # ---------------- MITRE Stage Prediction Head (6 stages) ----------------
        self.stage_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 6),
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass over a sequence of packet state observations.

        Args:
            x: Tensor of shape (batch, seq_len, 20)
            hx: Optional (hidden, cell) state tuple

        Returns:
            Dictionary with:
                - risk_prob: (batch, 1) probability P_packet(t)
                - future_states: (batch, horizon=5, 20)
                - stage_logits: (batch, 6)
                - stage_probs: (batch, 6)
                - latent: (batch, 64)
        """
        _, (hidden, cell) = self.lstm(x, hx)
        latent = self.shared(hidden[-1])

        future_flat = self.future_head(latent)
        future_states = future_flat.view(-1, self.prediction_horizon, self.input_size)

        risk = self.risk_head(latent)
        stage_logits = self.stage_head(latent)
        stage_probs = F.softmax(stage_logits, dim=-1)

        return {
            "risk_prob": risk.squeeze(-1),
            "future_states": future_states,
            "stage_logits": stage_logits,
            "stage_probs": stage_probs,
            "latent": latent,
            "hidden_state": hidden,
            "cell_state": cell,
        }

    def forward_tuple(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible tuple interface matching world_model_lstm scripts."""
        out = self.forward(x)
        return out["future_states"], out["risk_prob"].unsqueeze(-1), out["stage_logits"]

    def imagine(
        self,
        current_sequence: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the multi-step forward state forecast and MITRE stage trajectory."""
        self.eval()
        if current_sequence.ndim == 2:
            current_sequence = current_sequence.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(current_sequence)
            future_states = out["future_states"][0].cpu().numpy()  # (5, 20)
            risk = float(out["risk_prob"][0].cpu().item())
            stage_probs = out["stage_probs"][0].cpu().numpy()     # (6,)
            predicted_stage_id = int(np.argmax(stage_probs))

        timeline = []
        k_steps = horizon or self.prediction_horizon
        for k in range(k_steps):
            timeline.append({
                "step": k + 1,
                "lookahead": f"+{k+1} window",
                "predicted_stage_id": predicted_stage_id,
                "predicted_stage_name": PACKET_MITRE_STAGES[predicted_stage_id],
                "risk_prob": round(risk, 4),
            })

        return {
            "horizon": k_steps,
            "risk_prob": round(risk, 4),
            "predicted_stage_id": predicted_stage_id,
            "predicted_stage_name": PACKET_MITRE_STAGES[predicted_stage_id],
            "stage_probs": [round(float(p), 4) for p in stage_probs],
            "future_states": future_states.tolist(),
            "forecast_timeline": timeline,
        }
