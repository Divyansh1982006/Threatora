"""What-If Action Counterfactual Simulation Blueprint."""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, current_app

from src.config import SAMPLES_DIR
from src.simulation import WhatIfSimulationEngine
from server.blueprints.auth import login_required, permission_required

simulation_bp = Blueprint("simulation", __name__)


@simulation_bp.route("/api/v1/simulate/actions", methods=["GET"])
def get_supported_actions():
    """Lists all available counterfactual defensive actions and their descriptions."""
    sim_engine = current_app.extensions.get("simulation_engine")
    if sim_engine is None:
        return jsonify({"status": "error", "message": "ML engines are still initializing. Retry in ~30 seconds."}), 503
    actions = [
        {
            "key": k,
            "name": v["name"],
            "description": v["description"],
            "features_impacted": list(v["feature_dampeners"].keys())
        }
        for k, v in sim_engine.SUPPORTED_ACTIONS.items()
    ]
    return jsonify({"status": "success", "count": len(actions), "actions": actions})


@simulation_bp.route("/api/v1/simulate", methods=["POST"])
@login_required
@permission_required("can_simulate")
def simulate_action():
    """Executes a What-If counterfactual simulation on a host or telemetry window."""
    sim_engine = current_app.extensions.get("simulation_engine")
    inference_engine = current_app.extensions.get("inference_engine")
    if sim_engine is None or inference_engine is None:
        return jsonify({"status": "error", "message": "ML engines are still initializing. Retry in ~30 seconds."}), 503

    data = request.get_json(silent=True) or {}
    action_key = data.get("action", "BLOCK_MANAGEMENT_PORTS").upper()
    target_ip = data.get("target_ip")
    horizon = int(data.get("horizon", 10))

    if action_key not in sim_engine.SUPPORTED_ACTIONS:
        return jsonify({
            "status": "error",
            "message": f"Invalid action '{action_key}'. Supported: {list(sim_engine.SUPPORTED_ACTIONS.keys())}"
        }), 400

    # Obtain canonical (20, 12) context window from request, active telemetry, or sample data
    context_cells = None
    if "window" in data or "active_window" in data:
        context_cells = np.array(data.get("window") or data.get("active_window"), dtype=np.float32)
    elif current_app.extensions.get("latest_results") and "s_t_windows" in current_app.extensions["latest_results"]:
        windows = current_app.extensions["latest_results"]["s_t_windows"]
        if len(windows) > 0:
            context_cells = np.array(windows[-1], dtype=np.float32)

    if context_cells is None:
        pipeline = current_app.extensions.get("telemetry_pipeline")
        sample_path = SAMPLES_DIR / "sample_traffic.csv"
        if pipeline and sample_path.exists():
            raw_features, timestamps, _, _ = pipeline.parse_csv_stream(sample_path)
            res = pipeline.run_inference_on_features(raw_features, timestamps)
            if res.get("s_t_windows"):
                context_cells = np.array(res["s_t_windows"][-1], dtype=np.float32)

    if context_cells is None:
        # Fallback default baseline window of shape (20, 16)
        context_cells = np.zeros((20, 16), dtype=np.float32)
        context_cells[:, 2] = 1200.0  # packet_rate
        context_cells[:, 7] = 0.25    # syn_ratio
        context_cells[:, 10] = 1.0   # is_privileged_port

    # Ensure shape has 20 timesteps and 16 canonical features
    if context_cells.ndim == 3:
        context_cells = context_cells[0]
    if context_cells.shape[0] < 20:
        pad = np.repeat(context_cells[:1], 20 - context_cells.shape[0], axis=0)
        context_cells = np.vstack([pad, context_cells])
    elif context_cells.shape[0] > 20:
        context_cells = context_cells[-20:]

    if context_cells.shape[1] < 16:
        pad_w = 16 - context_cells.shape[1]
        context_cells = np.pad(context_cells, ((0, 0), (0, pad_w)), mode="constant", constant_values=0.0)
    elif context_cells.shape[1] > 16:
        context_cells = context_cells[:, :16]

    selected_ip = target_ip or "192.168.1.105"

    # Run counterfactual simulation via ONNX Runtime CPU execution
    result = sim_engine.simulate_action(
        context_cells=context_cells,
        action_key=action_key,
        horizon=horizon,
    )
    result["target_ip"] = selected_ip

    return jsonify(result)
