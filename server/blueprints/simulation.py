"""What-If Action Counterfactual Simulation Blueprint."""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, current_app

from src.config import SAMPLES_DIR, SEQUENCE_LENGTH
from src.features.windows import build_host_windows_from_flows, FeatureScaler
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

    # Obtain flow context: from request payload or fallback to sample attack traffic
    if "flows" in data and isinstance(data["flows"], list):
        df = pd.DataFrame(data["flows"])
    else:
        sample_path = SAMPLES_DIR / "sample_traffic.csv"
        df = pd.read_csv(sample_path)

    X_cells, _, _, meta = build_host_windows_from_flows(df)
    if len(X_cells) == 0:
        return jsonify({"status": "error", "message": "No valid 60s state windows constructed."}), 400

    scaler = FeatureScaler.load()
    X_scaled = scaler.transform(X_cells)

    # Filter by target host if specified
    unique_hosts = sorted(list(set(m[0] for m in meta)))
    if target_ip and target_ip in unique_hosts:
        selected_ip = target_ip
    else:
        selected_ip = unique_hosts[0]

    host_indices = [idx for idx, m in enumerate(meta) if m[0] == selected_ip]
    host_cells = X_scaled[host_indices]

    if len(host_cells) < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - len(host_cells)
        pad_head = np.repeat(host_cells[:1], pad_len, axis=0)
        context_cells = np.vstack([pad_head, host_cells])
    else:
        context_cells = host_cells[-SEQUENCE_LENGTH:]

    # Run counterfactual simulation
    result = sim_engine.simulate_action(
        context_cells=context_cells,
        action_key=action_key,
        horizon=horizon,
    )
    result["target_ip"] = selected_ip

    return jsonify(result)
