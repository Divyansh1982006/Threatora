"""Telemetry Ingestion & Inference Blueprint."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import pandas as pd

from src.config import SAMPLES_DIR
from src.prepare_data import generate_sample_attack_traffic
from server.blueprints.auth import login_required, permission_required

telemetry_bp = Blueprint("telemetry", __name__)

sample_path = SAMPLES_DIR / "sample_traffic.csv"
if not sample_path.exists():
    generate_sample_attack_traffic(sample_path)


@telemetry_bp.route("/api/v1/telemetry", methods=["POST", "GET"])
def api_v1_telemetry():
    """REST API endpoint for real-time telemetry stream ingestion and analysis."""
    engine = current_app.extensions["inference_engine"]
    mitigation_engine = current_app.extensions["mitigation_engine"]

    if request.method == "GET":
        if not sample_path.exists():
            generate_sample_attack_traffic(sample_path)
        df = pd.read_csv(sample_path)
        res = engine.process_traffic_dataframe(df)
        playbooks = mitigation_engine.evaluate_and_generate_playbooks(res)
        res["playbooks"] = playbooks
        return jsonify(res)

    if request.is_json:
        data = request.get_json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "flows" in data:
            df = pd.DataFrame(data["flows"])
        else:
            return jsonify({"status": "error", "message": "Expected JSON array of flow records or {'flows': [...]}"}), 400
        res = engine.process_traffic_dataframe(df)
    elif "file" in request.files:
        return upload_file()
    else:
        df = pd.read_csv(sample_path)
        res = engine.process_traffic_dataframe(df)

    playbooks = mitigation_engine.evaluate_and_generate_playbooks(res)
    res["playbooks"] = playbooks
    return jsonify(res)


@telemetry_bp.route("/api/demo", methods=["GET"])
def run_demo():
    """Runs instant inference and mitigation synthesis on multi-stage attack scenario."""
    engine = current_app.extensions["inference_engine"]
    mitigation_engine = current_app.extensions["mitigation_engine"]

    if not sample_path.exists():
        generate_sample_attack_traffic(sample_path)
    df = pd.read_csv(sample_path)
    res = engine.process_traffic_dataframe(df)

    playbooks = mitigation_engine.evaluate_and_generate_playbooks(res)
    res["playbooks"] = playbooks
    return jsonify(res)


@telemetry_bp.route("/api/upload", methods=["POST"])
@login_required
@permission_required("can_upload")
def upload_file():
    """Accepts PCAP or CSV flow file, runs forward simulation and synthesizes playbooks."""
    engine = current_app.extensions["inference_engine"]
    mitigation_engine = current_app.extensions["mitigation_engine"]

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if suffix in (".pcap", ".pcapng", ".cap"):
            res = engine.process_pcap(tmp_path)
        elif suffix in (".csv", ".txt", ".binetflow"):
            df = pd.read_csv(tmp_path)
            res = engine.process_traffic_dataframe(df)
        else:
            return jsonify({"status": "error", "message": f"Unsupported format '{suffix}'."}), 400

        playbooks = mitigation_engine.evaluate_and_generate_playbooks(res)
        res["playbooks"] = playbooks
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
