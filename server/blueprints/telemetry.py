"""Telemetry Ingestion & Multi-Modal Inference Blueprint."""

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
        res = engine.process_flow_csv(df)
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
        res = engine.process_flow_csv(df)
    elif "file" in request.files or "flow_file" in request.files or "packet_file" in request.files or "pcap_file" in request.files:
        return upload_file()
    else:
        df = pd.read_csv(sample_path)
        res = engine.process_flow_csv(df)

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
    res = engine.process_flow_csv(df)

    playbooks = mitigation_engine.evaluate_and_generate_playbooks(res)
    res["playbooks"] = playbooks
    return jsonify(res)


@telemetry_bp.route("/api/upload", methods=["POST"])
@login_required
@permission_required("can_upload")
def upload_file():
    """Accepts single (CSV/PCAP) or dual-modal (Flow CSV + PCAP) files, runs world model & fusion."""
    engine = current_app.extensions["inference_engine"]
    mitigation_engine = current_app.extensions["mitigation_engine"]

    # Check for dual-file upload: flow_file + packet_file (or pcap_file)
    has_dual = ("flow_file" in request.files) and ("packet_file" in request.files or "pcap_file" in request.files)
    
    saved_tmp_files = []

    try:
        if has_dual:
            flow_file = request.files["flow_file"]
            packet_file = request.files.get("packet_file", request.files.get("pcap_file"))

            flow_suffix = Path(secure_filename(flow_file.filename)).suffix.lower()
            packet_suffix = Path(secure_filename(packet_file.filename)).suffix.lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=flow_suffix) as f_tmp:
                flow_file.save(f_tmp.name)
                saved_tmp_files.append(f_tmp.name)
                flow_tmp_path = f_tmp.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=packet_suffix) as p_tmp:
                packet_file.save(p_tmp.name)
                saved_tmp_files.append(p_tmp.name)
                packet_tmp_path = p_tmp.name

            res = engine.process_network_input(flow_input=flow_tmp_path, packet_input=packet_tmp_path)
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"status": "error", "message": "Empty filename."}), 400

            filename = secure_filename(file.filename)
            suffix = Path(filename).suffix.lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                saved_tmp_files.append(tmp.name)
                tmp_path = tmp.name

            if suffix in (".pcap", ".pcapng", ".cap"):
                res = engine.process_pcap(tmp_path)
            elif suffix in (".csv", ".txt", ".binetflow"):
                res = engine.process_flow_csv(tmp_path)
            else:
                return jsonify({"status": "error", "message": f"Unsupported format '{suffix}'."}), 400
        else:
            return jsonify({"status": "error", "message": "No file uploaded."}), 400

        playbooks = mitigation_engine.evaluate_and_generate_playbooks(res)
        res["playbooks"] = playbooks
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        for tmp_path in saved_tmp_files:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
