"""What-If Action Simulation Engine for Threatora World Model (NTRO PS 26153).

Operational Role:
  - Injects counterfactual defensive actions into the 12 canonical continuous state features.
  - Re-unrolls the Temporal Transformer World Model via ONNX Runtime CPU execution.
  - Models future network trajectories and risk curves under hypothetical interventions.
  - Quantifies risk reduction: ΔRisk = Risk(Unmitigated) - Risk(Intervention).
  - Empowers SOC operators to validate firewall/isolation impact before deploying commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort

from .adapters.dataset_adapter import CANONICAL_SLOTS
from .mitre import STAGE_COLORS, STAGE_NAMES

# Feature index mapping for fast numpy slicing across 12 canonical slots
FEAT_IDX = {col: i for i, col in enumerate(CANONICAL_SLOTS)}


class WhatIfSimulationEngine:
    """Simulates counterfactual network states and risk trajectories under defensive actions."""

    SUPPORTED_ACTIONS = {
        "ISOLATE_HOST": {
            "name": "Host Isolation (Air-gap)",
            "description": "Sever all inbound and outbound host routing (egress/inbound -> 0, bytes -> 0).",
            "feature_dampeners": {
                "byte_ratio": 0.05,
                "packet_rate": 0.05,
                "is_privileged_port": 0.0,
                "payload_entropy": 0.05,
                "fwd_bwd_packet_ratio": 0.1,
                "payload_bytes_mean": 0.05,
                "tcp_rst_ratio": 0.0,
            },
        },
        "BLOCK_MANAGEMENT_PORTS": {
            "name": "Block Ingress Management Ports (22/3389/445)",
            "description": "Block SSH, RDP, and SMB lateral exploration and brute-forcing.",
            "feature_dampeners": {
                "is_privileged_port": 0.0,
                "tcp_window_norm": 0.2,
                "tcp_rst_ratio": 0.1,
            },
        },
        "RATE_LIMIT_SYN": {
            "name": "Rate-Limit TCP SYN Probing",
            "description": "Filter aggressive TCP SYN probing bursts on perimeter ingress.",
            "feature_dampeners": {
                "tcp_syn_ratio": 0.05,
                "packet_rate": 0.3,
                "tcp_rst_ratio": 0.1,
            },
        },
        "SINKHOLE_C2_DNS": {
            "name": "DNS Sinkhole & C2 Severance",
            "description": "Null-route malicious C2 heartbeat domains and disrupt periodic beaconing.",
            "feature_dampeners": {
                "payload_entropy": 0.1,
                "byte_ratio": 0.2,
                "iat_std": 2.0,
                "payload_bytes_mean": 0.2,
            },
        },
        "THROTTLE_EGRESS": {
            "name": "Throttle Outbound Bandwidth",
            "description": "Sever high-volume data exfiltration channels and rate-limit egress bursts.",
            "feature_dampeners": {
                "byte_ratio": 0.1,
                "packet_rate": 0.2,
                "payload_bytes_mean": 0.2,
                "fwd_bwd_packet_ratio": 0.3,
            },
        },
    }

    def __init__(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        model: Optional[Any] = None,
        device: Optional[Any] = None,
        **kwargs: Any,
    ):
        repo_root = Path(__file__).resolve().parent.parent
        self.onnx_path = Path(onnx_path) if onnx_path else repo_root / "models" / "threatora_transformer.onnx"

        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"Threatora Transformer ONNX model not found at: {self.onnx_path}. "
                "Ensure models/threatora_transformer.onnx exists before initializing simulation engine."
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def _rollout_k_steps(
        self, initial_state: np.ndarray, horizon: int = 5
    ) -> Tuple[List[float], List[np.ndarray]]:
        """Autoregressively rolls out state dynamics and threat risk over horizon k on CPU."""
        cur_state = np.array(initial_state, dtype=np.float32)
        if cur_state.ndim == 2:
            cur_state = np.expand_dims(cur_state, axis=0)

        probs: List[float] = []
        states: List[np.ndarray] = []

        for _ in range(horizon):
            outs = self.session.run(
                None, {"input_s_t": cur_state}
            )
            pred_s_next = outs[0]
            primary_logit = outs[1]
            # Sigmoid activation on primary threat logit
            logit_val = float(primary_logit[0, 0])
            p = float(1.0 / (1.0 + np.exp(-logit_val)))
            probs.append(p)
            states.append(pred_s_next[0])
            # Autoregressive step: next input state is the reconstructed state tensor
            cur_state = pred_s_next

        return probs, states

    def simulate_action(
        self,
        context_cells: np.ndarray,
        action_key: str,
        horizon: int = 5,
    ) -> Dict[str, Any]:
        """Runs a side-by-side counterfactual simulation comparing baseline vs action via ONNX CPU execution."""
        if action_key not in self.SUPPORTED_ACTIONS:
            raise ValueError(
                f"Unsupported action '{action_key}'. Choose from: {list(self.SUPPORTED_ACTIONS.keys())}"
            )

        action_meta = self.SUPPORTED_ACTIONS[action_key]
        context = np.array(context_cells, dtype=np.float32)
        if context.ndim == 3:
            context = context.squeeze(0)

        # 1. Baseline Simulation (Unmitigated) via ONNX Runtime
        base_probs, base_states = self._rollout_k_steps(context, horizon=horizon)

        # 2. Perturb observation context to model action intervention on 12 canonical features
        perturbed_cells = np.copy(context)
        trailing_bins = min(4, len(perturbed_cells))
        for feat, multiplier in action_meta["feature_dampeners"].items():
            if feat in FEAT_IDX:
                idx = FEAT_IDX[feat]
                perturbed_cells[-trailing_bins:, idx] *= multiplier

        # 3. Counterfactual Simulation (Action Applied) via ONNX Runtime
        sim_probs, sim_states = self._rollout_k_steps(perturbed_cells, horizon=horizon)

        # 4. Compute Metrics & Deltas
        mean_base_risk = float(np.mean(base_probs))
        mean_sim_risk = float(np.mean(sim_probs))
        delta_mean_risk = max(mean_base_risk - mean_sim_risk, 0.0)
        pct_risk_reduction = (
            (delta_mean_risk / max(mean_base_risk, 1e-4)) * 100.0
            if mean_base_risk > 0.01
            else 0.0
        )

        peak_base_risk = float(np.max(base_probs))
        peak_sim_risk = float(np.max(sim_probs))
        delta_peak_risk = max(peak_base_risk - peak_sim_risk, 0.0)

        # Format comparison timeline
        timeline_comparison = []
        for k in range(horizon):
            timeline_comparison.append({
                "step": k + 1,
                "minute": f"+{(k + 1) * 0.5:.1f}s",
                "baseline_risk": round(base_probs[k], 4),
                "simulated_risk": round(sim_probs[k], 4),
                "delta_risk": round(base_probs[k] - sim_probs[k], 4),
                "baseline_stage": "Elevated Threat" if base_probs[k] >= 0.5 else "Benign Baseline",
                "baseline_color": "#ef4444" if base_probs[k] >= 0.5 else "#10b981",
                "simulated_stage": "Elevated Threat" if sim_probs[k] >= 0.5 else "Benign Baseline",
                "simulated_color": "#ef4444" if sim_probs[k] >= 0.5 else "#10b981",
            })

        # Synthesize tactical recommendation
        if pct_risk_reduction >= 40.0:
            effectiveness = "HIGHLY_EFFECTIVE"
            verdict = (
                f"Action '{action_meta['name']}' demonstrates high defensive leverage: "
                f"modeled {pct_risk_reduction:.1f}% risk reduction across the {horizon}-step horizon. "
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
            "timeline": timeline_comparison,
        }
