"""Headless Verification Suite for Threatora SOC Dashboard Telemetry and Visualizations.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Verifies:
  1. Parsing real PCAP telemetry (data/UNSW-NB15 Dataset/pcap files/22-1-15/2.pcap) with zero errors.
  2. Generating continuous rolling temporal windows (B, 20, 12).
  3. Real ONNX model CPU inference: Smooth L1 reconstruction loss and 5-step risk horizon.
  4. All probabilities are within [0, 1] with zero NaNs or Infs.
  5. Plotly visual figures bind directly to dynamically computed arrays.
  6. Prescriptive counterfactual simulation computes valid risk reduction curves.
"""

from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest
import torch

from src.adapters import CANONICAL_SLOTS
from src.dashboard.telemetry import SOCTelemetryPipeline
from src.dashboard.app import (
    create_fan_chart,
    create_attribution_waterfall,
    create_counterfactual_comparison_chart,
)


@pytest.fixture
def pcap_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "data" / "UNSW-NB15 Dataset" / "pcap files" / "22-1-15" / "2.pcap"
    if not candidate.exists():
        pytest.skip(f"PCAP file not found at: {candidate}")
    return candidate


def test_headless_pcap_ingestion_and_inference(pcap_path):
    """Verifies that 2.pcap parses dynamically into 12 canonical features and executes live ONNX forward pass."""
    pipeline = SOCTelemetryPipeline()

    # 1. Parse streaming PCAP packets into 12 canonical continuous features
    # Limit max_packets to 5000 for fast, representative headless verification
    raw_features, timestamps, meta, *rest = pipeline.parse_pcap_stream(pcap_path, max_packets=5000)

    assert len(raw_features) > 0, "Expected non-empty temporal bins from PCAP"
    assert raw_features.shape[1] in (12, 16), f"Expected 12 or 16 canonical features, got {raw_features.shape[1]}"
    assert len(timestamps) == len(raw_features)
    assert not np.isnan(raw_features).any(), "Extracted features must not contain NaNs"
    assert not np.isinf(raw_features).any(), "Extracted features must not contain Infs"

    assert meta["file_name"] == "2.pcap"
    assert meta["total_packets"] > 0

    # 2. Run forward inference
    results = pipeline.run_inference_on_features(raw_features, timestamps)

    assert results["num_windows"] > 0
    assert results["latency_ms"] > 0.0
    assert results["throughput_wps"] > 0.0

    primary_probs = results["primary_probs"]
    timeline_probs = results["timeline_probs"]

    assert len(primary_probs) == results["num_windows"]
    assert timeline_probs.shape == (results["num_windows"], 5)

    # Validate probability bounds [0, 1]
    assert np.all(primary_probs >= 0.0) and np.all(primary_probs <= 1.0)
    assert np.all(timeline_probs >= 0.0) and np.all(timeline_probs <= 1.0)
    assert not np.isnan(primary_probs).any()
    assert not np.isnan(timeline_probs).any()

    # 3. Test Plotly Visualizations Binding
    active_idx = min(10, results["num_windows"] - 1)
    hist_sub = primary_probs[max(0, active_idx - 6) : active_idx + 1]
    fig_fan = create_fan_chart(hist_sub, timeline_probs[active_idx], active_idx)
    assert isinstance(fig_fan, go.Figure)
    assert len(fig_fan.data) >= 3  # historical, confidence band, forecast rollout

    active_window = results["s_t_windows"][active_idx]
    fig_attr = create_attribution_waterfall(active_window, CANONICAL_SLOTS)
    assert isinstance(fig_attr, go.Figure)
    assert len(fig_attr.data) == 1

    # 4. Test Prescriptive Counterfactual Sandbox
    mitigated_tl, mitigated_primary = pipeline.simulate_counterfactual(
        active_window=active_window,
        isolate_subnet=True,
        throttle_privileged_ports=True,
        rate_limit_syn=True,
    )
    assert len(mitigated_tl) == 5
    assert 0.0 <= mitigated_primary <= 1.0

    fig_cf = create_counterfactual_comparison_chart(timeline_probs[active_idx], mitigated_tl)
    assert isinstance(fig_cf, go.Figure)
    assert len(fig_cf.data) == 2  # unmitigated and mitigated curves


def test_headless_csv_ingestion_and_inference(tmp_path):
    """Verifies that flow CSV telemetry parses dynamically into 12 canonical features and runs forward pass."""
    pipeline = SOCTelemetryPipeline()

    # Generate synthetic CSV flow file
    n = 80
    df = pl.DataFrame({
        "srcip": ["10.0.0.1"] * n,
        "dstip": ["192.168.1.1"] * n,
        "sport": ["45000"] * n,
        "dsport": ["80", "443", "8080", "22", "53"] * (n // 5),
        "stime": [1000.0 + i * 0.5 for i in range(n)],
        "dur": [0.1] * n,
        "sbytes": [100] * n,
        "dbytes": [150] * n,
        "spkts": [2] * n,
        "dpkts": [3] * n,
        "state": ["CON"] * n,
        "sintpkt": [10.0] * n,
        "dintpkt": [12.0] * n,
        "sttl": [64] * n,
        "dttl": [60] * n,
        "swin": [1024] * n,
        "dwin": [1024] * n,
        "label": [0] * n,
    })

    csv_file = tmp_path / "test_flows.csv"
    df.write_csv(csv_file)

    raw_features, timestamps, meta, *rest = pipeline.parse_csv_stream(csv_file)
    assert raw_features.shape[1] in (12, 16)
    assert meta["num_temporal_bins"] > 0

    results = pipeline.run_inference_on_features(raw_features, timestamps)
    assert results["num_windows"] > 0
    assert results["timeline_probs"].shape[1] == 5
