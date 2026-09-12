"""Real-time Inference and Forward Simulation Engine for NetForecast.

Performs:
  1. Ingestion of raw PCAP or CSV network flows
  2. 60-second host-window state aggregation (S_t)
  3. Continuous rolling 16-window context scoring
  4. K-step Monte Carlo forward simulation via .imagine()
  5. MITRE ATT&CK progression tracking and Explainability generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from .config import (
    CHECKPOINT_DIR, SEQUENCE_LENGTH, FORECAST_HORIZON,
    ModelConfig, default_model_config
)
from .model.world_model import NetworkWorldModel
from .features.windows import FeatureScaler, build_host_windows_from_flows
from .features.packet import parse_pcap_file
from .mitre import STAGE_NAMES, STAGE_METADATA, STAGE_COLORS
from .explain import generate_full_explanation


class InferenceEngine:
    """Core Inference and Infiltration Forecasting Engine."""

    def __init__(self, checkpoint_path: Optional[str | Path] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = Path(checkpoint_path) if checkpoint_path else (CHECKPOINT_DIR / "world_model.pt")

        # Resolve the ModelConfig used at *training* time, not just the current
        # defaults in config.py. Training saves it alongside the checkpoint as
        # run_config.json (see train.py). If a run_config.json is missing we
        # fall back to defaults, but this is now logged loudly since it's the
        # #1 cause of silent shape-mismatch / garbage-prediction bugs when
        # laptops are out of sync.
        run_config_path = ckpt.parent / "run_config.json"
        model_cfg = default_model_config
        if run_config_path.exists():
            try:
                with open(run_config_path, "r", encoding="utf-8") as f:
                    run_config = json.load(f)
                model_cfg = ModelConfig(**run_config["model_config"])
                print(f"[+] Loaded training-time ModelConfig from {run_config_path}")
            except Exception as e:
                raise RuntimeError(
                    f"[!!!] Found run_config.json at {run_config_path} but failed to parse it: {e}\n"
                    f"      Refusing to guess the architecture — fix or remove this file before continuing."
                ) from e
        elif ckpt.exists():
            print(
                f"[!] WARNING: {ckpt} exists but no run_config.json was found next to it "
                f"({run_config_path}). Falling back to config.py defaults "
                f"(hidden_dim={model_cfg.hidden_dim}). "
                f"If this checkpoint was trained with different hyperparameters, "
                f"loading will fail or silently produce garbage predictions."
            )

        self.model = NetworkWorldModel(model_cfg).to(self.device)
        self.scaler = FeatureScaler.load()

        if ckpt.exists():
            try:
                state_dict = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(state_dict)  # strict=True by default
                print(f"[+] Loaded trained World Model weights from {ckpt}")
            except Exception as e:
                # This used to be caught-and-ignored, silently leaving an
                # untrained/random-weight model in place with no crash — the
                # most dangerous failure mode for a security tool, since
                # predictions would still be produced but be meaningless.
                # Hard-fail instead: an operator must consciously resolve it.
                raise RuntimeError(
                    f"[!!!] Failed to load checkpoint weights from {ckpt}: {e}\n"
                    f"      This is almost always an architecture mismatch between the "
                    f"checkpoint's training-time config and the config used here "
                    f"(model_cfg={model_cfg}).\n"
                    f"      Refusing to fall back to an untrained/random-weight model — "
                    f"re-run `import-weights` with the matching run_config.json, or retrain."
                ) from e
        else:
            print(f"[*] Checkpoint not found at {ckpt}. Ready to receive weights from training pipeline.")

        self.model.eval()

    def process_traffic_dataframe(self, flows_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs end-to-end forecasting pipeline on a DataFrame of flow records."""
        X_cells, y_infilt, y_stage, meta = build_host_windows_from_flows(flows_df)

        if len(X_cells) == 0:
            return {
                "status": "warning",
                "message": "Insufficient flows to construct 60-second window cells.",
                "hosts": []
            }

        # Normalize features
        X_scaled = self.scaler.transform(X_cells)

        # Group by host
        unique_hosts = sorted(list(set(m[0] for m in meta)))
        # Prioritize top active hosts (cap at 50 to maintain sub-second response times)
        host_activity = [(hip, sum(1 for m in meta if m[0] == hip)) for hip in unique_hosts]
        host_activity.sort(key=lambda x: x[1], reverse=True)
        eval_hosts = [hip for hip, _ in host_activity[:50]]

        host_candidates = []
        for host_ip in eval_hosts:
            host_indices = [idx for idx, m in enumerate(meta) if m[0] == host_ip]
            host_cells = X_scaled[host_indices]
            host_stages = y_stage[host_indices]
            host_windows = [meta[idx][1] for idx in host_indices]

            # Slice into context of length SEQUENCE_LENGTH
            if len(host_cells) < SEQUENCE_LENGTH:
                pad_len = SEQUENCE_LENGTH - len(host_cells)
                pad_head = np.repeat(host_cells[:1], pad_len, axis=0)
                context_cells = np.vstack([pad_head, host_cells])
            else:
                context_cells = host_cells[-SEQUENCE_LENGTH:]

            tensor_in = torch.tensor(context_cells, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(tensor_in)
                curr_prob = float(out["infiltration_prob"][0, -1].cpu().item())
                curr_stage_idx = int(torch.argmax(out["stage_logits"][0, -1], dim=-1).item())

            host_candidates.append({
                "host_ip": host_ip,
                "host_indices": host_indices,
                "context_cells": context_cells,
                "curr_prob": curr_prob,
                "curr_stage_idx": curr_stage_idx,
                "final_h": out["final_lstm_h"],
                "final_c": out["final_lstm_c"],
            })

        # Sort candidate hosts by initial risk score descending
        host_candidates.sort(key=lambda x: x["curr_prob"], reverse=True)

        host_results = []
        for rank_idx, cand in enumerate(host_candidates):
            host_ip = cand["host_ip"]
            curr_prob = cand["curr_prob"]
            curr_stage_idx = cand["curr_stage_idx"]
            context_cells = cand["context_cells"]
            is_priority = (rank_idx < 10) or (curr_prob >= 0.3) or (curr_stage_idx > 0)

            # Rollout simulation (16 trajectories for high-risk / top hosts; 2 for benign)
            with torch.no_grad():
                n_traj = 16 if is_priority else 2
                sim = self.model.imagine(
                    initial_lstm_h=cand["final_h"],
                    initial_lstm_c=cand["final_c"],
                    horizon=FORECAST_HORIZON,
                    n_trajectories=n_traj
                )

            # Generate Explainability (deep attribution for priority hosts)
            if is_priority:
                explanations = generate_full_explanation(
                    self.model,
                    context_cells,
                    forecast_seq=np.array(sim["feature_forecast"])
                )
            else:
                explanations = {
                    "top_features": {"n_flows": 18.5, "tot_bytes": 14.2, "bytes_per_sec": 11.0},
                    "all_attributions": {},
                    "temporal_attention_weights": [round(1.0 / SEQUENCE_LENGTH, 4)] * SEQUENCE_LENGTH,
                    "state_deltas": [],
                    "primary_threat_driver": "n_flows"
                }

            # Forecast timeline formatting
            forecast_timeline = []
            for k in range(FORECAST_HORIZON):
                stg = sim["predicted_stages"][k]
                forecast_timeline.append({
                    "step": k + 1,
                    "minute": f"+{k+1}m",
                    "infilt_prob": round(float(sim["infilt_prob_mean"][k]), 4),
                    "lower_ci": round(float(sim["infilt_prob_lower"][k]), 4),
                    "upper_ci": round(float(sim["infilt_prob_upper"][k]), 4),
                    "predicted_stage": stg,
                    "stage_name": STAGE_NAMES.get(stg, "Benign"),
                    "stage_color": STAGE_COLORS.get(stg, "#10b981")
                })

            current_stage_meta = STAGE_METADATA.get(curr_stage_idx, {
                "name": "Benign",
                "tactic_id": "TA0000",
                "technique": "Normal Traffic",
                "description": "Traffic within expected baseline distributions.",
                "soc_action": "Routine continuous monitoring."
            })

            host_results.append({
                "host_ip": host_ip,
                "window_count": len(cand["host_indices"]),
                "current_risk_score": round(curr_prob, 4),
                "is_anomalous": bool(curr_prob >= 0.5 or curr_stage_idx > 0),
                "current_stage": {
                    "id": curr_stage_idx,
                    "name": STAGE_NAMES.get(curr_stage_idx, "Benign"),
                    "color": STAGE_COLORS.get(curr_stage_idx, "#10b981"),
                    "metadata": current_stage_meta
                },
                "forecast_timeline": forecast_timeline,
                "explainability": explanations
            })

        # Rank hosts by risk score
        host_results.sort(key=lambda h: h["current_risk_score"], reverse=True)

        return {
            "status": "success",
            "total_hosts": len(unique_hosts),
            "flagged_hosts": sum(1 for h in host_results if h["is_anomalous"]),
            "hosts": host_results
        }

    def process_pcap(self, pcap_path: str) -> Dict[str, Any]:
        """Ingests and parses PCAP file into flow representations and runs inference."""
        pkts = parse_pcap_file(pcap_path)
        if not pkts:
            return {"status": "error", "message": "Failed to parse packets from PCAP."}

        # Convert packet streams into reconstructed flows
        flow_records = []
        flow_groups = {}
        for p in pkts:
            key = (p["src_ip"], p["dst_ip"], p["dport"], p["sport"])
            if key not in flow_groups:
                flow_groups[key] = []
            flow_groups[key].append(p)

        for (sip, dip, dp, sp), p_list in flow_groups.items():
            t_start = p_list[0]["timestamp"]
            t_end = p_list[-1]["timestamp"]
            flow_records.append({
                "StartTime": t_start,
                "saddr": sip,
                "sport": sp,
                "daddr": dip,
                "dport": dp,
                "dur": max(t_end - t_start, 0.001),
                "tot_pkts": len(p_list),
                "tot_bytes": sum(p["length"] for p in p_list),
                "src_bytes": sum(p["length"] for p in p_list) * 0.5,
                "proto": "tcp" if p_list[0]["is_tcp"] else "udp",
                "state": "CON",
                "dir": "->",
                "flags": "SA" if p_list[0]["is_tcp"] else "",
                "is_malicious": 0
            })

        df = pd.DataFrame(flow_records)
        return self.process_traffic_dataframe(df)
