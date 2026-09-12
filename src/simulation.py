"""What-If Action Simulation Engine for Threatora World Model (NTRO PS 26153).

Operational Role:
  - Injects counterfactual defensive actions into the latent network state.
  - Re-unrolls the Decoder LSTM (.imagine) to model future network trajectories under hypothetical interventions.
  - Quantifies risk reduction: ΔRisk = Risk(Unmitigated) - Risk(Intervention).
  - Empowers SOC operators to validate firewall/isolation impact before deploying commands.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from .config import (
    ALL_FEATURE_COLS, SEQUENCE_LENGTH, FORECAST_HORIZON,
    CHECKPOINT_DIR, ModelConfig
)
from .mitre import STAGE_NAMES, STAGE_COLORS
from .model.world_model import NetworkWorldModel


# Feature index mapping for fast tensor slicing
FEAT_IDX = {col: i for i, col in enumerate(ALL_FEATURE_COLS)}


class WhatIfSimulationEngine:
    """Simulates counterfactual network states under defensive actions."""

    SUPPORTED_ACTIONS = {
        "ISOLATE_HOST": {
            "name": "Host Isolation (Air-gap)",
            "description": "Sever all inbound and outbound host routing (egress/inbound -> 0, bytes -> 0).",
            "feature_dampeners": {
                "frac_outbound": 0.0,
                "egress_ratio": 0.0,
                "bytes_per_sec": 0.05,
                "pkts_per_sec": 0.05,
                "tot_bytes": 0.05,
                "tot_pkts": 0.05,
                "n_flows": 0.05,
            }
        },
        "BLOCK_MANAGEMENT_PORTS": {
            "name": "Block Ingress Management Ports (22/3389/445)",
            "description": "Block SSH, RDP, and SMB lateral exploration and brute-forcing.",
            "feature_dampeners": {
                "n_unique_dport": 0.2,
                "dport_entropy": 0.2,
                "packet_dport_entropy": 0.2,
                "sequential_portscan_score": 0.1,
                "frac_established": 0.3,
            }
        },
        "RATE_LIMIT_SYN": {
            "name": "Rate-Limit TCP SYN Probing",
            "description": "Filter aggressive TCP SYN probing bursts on perimeter ingress.",
            "feature_dampeners": {
                "frac_syn_only": 0.1,
                "pkts_per_sec": 0.3,
                "sequential_portscan_score": 0.1,
                "frac_reset": 0.2,
            }
        },
        "SINKHOLE_C2_DNS": {
            "name": "DNS Sinkhole & C2 Severance",
            "description": "Null-route malicious C2 heartbeat domains and terminate periodic DNS beacons.",
            "feature_dampeners": {
                "dns_query_count": 0.05,
                "beacon_regularity": 0.1,
                "frac_psh": 0.3,
            }
        },
        "THROTTLE_EGRESS": {
            "name": "Throttle Outbound Bandwidth",
            "description": "Sever high-volume data exfiltration channels and rate-limit egress bursts.",
            "feature_dampeners": {
                "src_bytes": 0.1,
                "bytes_per_sec": 0.15,
                "egress_ratio": 0.2,
                "tot_bytes": 0.2,
            }
        },
    }

    def __init__(self, model: Optional[NetworkWorldModel] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None and isinstance(model, NetworkWorldModel):
            self.model = model.to(self.device)
        else:
            self.model = NetworkWorldModel().to(self.device)
            ckpt_path = CHECKPOINT_DIR / "world_model.pt"
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    self.model.load_state_dict(state_dict)
                except Exception:
                    pass
        self.model.eval()

    def simulate_action(
        self,
        context_cells: np.ndarray,
        action_key: str,
        horizon: int = FORECAST_HORIZON,
        n_trajectories: int = 16,
    ) -> Dict[str, Any]:
        """Runs a side-by-side counterfactual simulation comparing baseline vs action."""
        if action_key not in self.SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action '{action_key}'. Choose from: {list(self.SUPPORTED_ACTIONS.keys())}")

        action_meta = self.SUPPORTED_ACTIONS[action_key]

        # 1. Baseline Simulation (Unmitigated)
        tensor_baseline = torch.tensor(context_cells, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_base = self.model(tensor_baseline)
            sim_base = self.model.imagine(
                initial_lstm_h=out_base["final_lstm_h"],
                initial_lstm_c=out_base["final_lstm_c"],
                horizon=horizon,
                n_trajectories=n_trajectories,
                mc_dropout=False
            )

        # 2. Perturb observation context to model action intervention
        perturbed_cells = np.copy(context_cells)
        # Apply intervention to the recent trailing context windows
        trailing_windows = min(4, len(perturbed_cells))
        for feat, multiplier in action_meta["feature_dampeners"].items():
            if feat in FEAT_IDX:
                idx = FEAT_IDX[feat]
                perturbed_cells[-trailing_windows:, idx] *= multiplier

        # 3. Counterfactual Simulation (Action Applied)
        tensor_sim = torch.tensor(perturbed_cells, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_sim = self.model(tensor_sim)
            sim_counterfactual = self.model.imagine(
                initial_lstm_h=out_sim["final_lstm_h"],
                initial_lstm_c=out_sim["final_lstm_c"],
                horizon=horizon,
                n_trajectories=n_trajectories,
                mc_dropout=False
            )

        # 4. Compute Metrics & Deltas
        base_probs = [float(p) for p in sim_base["infilt_prob_mean"]]
        sim_probs = [float(p) for p in sim_counterfactual["infilt_prob_mean"]]

        mean_base_risk = float(np.mean(base_probs))
        mean_sim_risk = float(np.mean(sim_probs))
        delta_mean_risk = max(mean_base_risk - mean_sim_risk, 0.0)
        pct_risk_reduction = (delta_mean_risk / max(mean_base_risk, 1e-4)) * 100.0

        peak_base_risk = float(np.max(base_probs))
        peak_sim_risk = float(np.max(sim_probs))
        delta_peak_risk = max(peak_base_risk - peak_sim_risk, 0.0)

        # Format comparison timeline
        timeline_comparison = []
        for k in range(horizon):
            b_stg = sim_base["predicted_stages"][k]
            s_stg = sim_counterfactual["predicted_stages"][k]
            timeline_comparison.append({
                "step": k + 1,
                "minute": f"+{k+1}m",
                "baseline_risk": round(base_probs[k], 4),
                "simulated_risk": round(sim_probs[k], 4),
                "delta_risk": round(base_probs[k] - sim_probs[k], 4),
                "baseline_stage": STAGE_NAMES.get(b_stg, "Benign"),
                "baseline_color": STAGE_COLORS.get(b_stg, "#10b981"),
                "simulated_stage": STAGE_NAMES.get(s_stg, "Benign"),
                "simulated_color": STAGE_COLORS.get(s_stg, "#10b981"),
            })

        # Synthesize tactical recommendation
        if pct_risk_reduction >= 40.0:
            effectiveness = "HIGHLY_EFFECTIVE"
            verdict = (
                f"Action '{action_meta['name']}' demonstrates high defensive leverage: "
                f"modeled {pct_risk_reduction:.1f}% risk reduction across the {horizon}-minute horizon. "
                f"Recommended for immediate deployment."
            )
        elif pct_risk_reduction >= 15.0:
            effectiveness = "MODERATELY_EFFECTIVE"
            verdict = (
                f"Action '{action_meta['name']}' dampens attack velocity by {pct_risk_reduction:.1f}%. "
                f"Combine with network segmentation for complete containment."
            )
        else:
            effectiveness = "LOW_IMPACT"
            verdict = (
                f"Action '{action_meta['name']}' yields nominal risk reduction ({pct_risk_reduction:.1f}%). "
                f"Attack vectors likely utilize distinct protocol channels."
            )

        return {
            "status": "success",
            "action_key": action_key,
            "action_name": action_meta["name"],
            "action_description": action_meta["description"],
            "horizon": horizon,
            "effectiveness": effectiveness,
            "tactical_verdict": verdict,
            "mean_baseline_risk": round(mean_base_risk, 4),
            "mean_simulated_risk": round(mean_sim_risk, 4),
            "pct_risk_reduction": round(pct_risk_reduction, 1),
            "peak_risk_reduction": round(delta_peak_risk, 4),
            "timeline": timeline_comparison
        }
