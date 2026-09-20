"""Unit tests for offline PCAP evaluation script (src/evaluate_pcap.py)."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.evaluate_pcap import (
    CORE_FEATURE_NAMES,
    MITRE_STAGE_NAMES,
    infer_mitre_stage,
    format_pcap_evaluation_table,
)


def test_core_features_list():
    assert len(CORE_FEATURE_NAMES) == 12
    assert CORE_FEATURE_NAMES[0] == "byte_rate"
    assert CORE_FEATURE_NAMES[1] == "packet_rate"
    assert CORE_FEATURE_NAMES[-1] == "mean_flow_duration"


def test_infer_mitre_stage():
    # Benign window when both immediate threat and horizon risk are low
    dummy_window = np.zeros((20, 12), dtype=np.float32)
    stage_id, stage_name, tech = infer_mitre_stage(
        threat_prob=0.1,
        horizon_max_risk=0.2,
        features_window=dummy_window,
    )
    assert stage_id == 0
    assert stage_name == "Benign"

    # Reconnaissance: elevated risk + high SYN ratio
    recon_window = np.zeros((20, 12), dtype=np.float32)
    recon_window[:, 5] = 0.6  # High syn_flag_ratio
    stage_id, stage_name, tech = infer_mitre_stage(
        threat_prob=0.6,
        horizon_max_risk=0.7,
        features_window=recon_window,
    )
    assert stage_id == 1
    assert stage_name == "Reconnaissance"

    # Exfiltration: elevated risk + high byte rate
    exfil_window = np.zeros((20, 12), dtype=np.float32)
    exfil_window[:, 0] = 600000.0  # High byte_rate
    stage_id, stage_name, tech = infer_mitre_stage(
        threat_prob=0.8,
        horizon_max_risk=0.9,
        features_window=exfil_window,
    )
    assert stage_id == 5
    assert stage_name == "Exfiltration"


def test_format_pcap_evaluation_table():
    sample_report = {
        "pcap_metadata": {
            "pcap_file": "2.pcap",
            "pcap_size_mb": 953.67,
            "total_packets": 1614980,
            "duration_seconds": 690.36,
        },
        "inference_benchmark": {
            "total_evaluated_windows": 1362,
            "cpu_latency_per_sample_ms": 0.150,
            "throughput_windows_per_sec": 6666.0,
            "mean_state_reconstruction_loss": 2.0943,
            "onnx_model_footprint_mb": 0.35,
        },
        "threat_forecasting_summary": {
            "threat_windows_count": 10,
            "threat_ratio_pct": 0.73,
            "mean_lead_time_to_compromise_sec": 47.0,
        },
        "mitre_attack_progression": {
            "stage_distribution": {
                "Reconnaissance": 2,
                "Initial Access": 3,
                "Command & Control": 1,
                "Exfiltration": 4,
            }
        },
    }
    table = format_pcap_evaluation_table(sample_report)
    assert "Threatora Temporal Transformer" in table
    assert "2.pcap" in table
    assert "State Transition Loss" in table
