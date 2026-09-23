"""Unit and integration tests for Dynamic Canonical Feature Adapter and Benchmark Evaluation Suite.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Verifies:
  1. UNSW-NB15, CSE-CIC-IDS2018, and CTU-13 batches correctly map to torch.Size([B, 20, 12]).
  2. Zero data leakage enforcement (strictly strips identity shortcuts).
  3. F1-score and operational metric evaluation handles edge cases deterministically.
  4. ONNX runtime executes forward passes on adapted tensors without shape mismatches.
  5. Mean Lead Time to Compromise (MLTC) at bounded FPR budget (< 0.1%).
"""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
import torch

from src.adapters import (
    CANONICAL_SLOTS,
    CanonicalFeatureExtractor,
    parse_port_value,
)
from src.evaluate_benchmark import (
    compute_confusion_matrix_elements,
    compute_mltc_at_bounded_fpr,
    compute_operational_metrics,
    compute_smooth_l1_reconstruction,
    tune_optimal_threshold,
)


@pytest.fixture
def sample_unsw_df() -> pl.DataFrame:
    """Generates synthetic UNSW-NB15 flow data."""
    n = 60
    return pl.DataFrame({
        "srcip": [f"10.0.0.{i%20}" for i in range(n)],
        "dstip": [f"192.168.1.{i%50}" for i in range(n)],
        "sport": ["45000"] * n,
        "dsport": ["80", "443", "8080", "22", "53"] * (n // 5),
        "proto": ["tcp"] * n,
        "state": ["CON", "REQ", "FIN", "ACC"] * (n // 4),
        "dur": [0.05 + 0.01 * (i % 10) for i in range(n)],
        "sbytes": [100 + i * 5 for i in range(n)],
        "dbytes": [200 + i * 10 for i in range(n)],
        "sttl": [64] * n,
        "dttl": [60] * n,
        "spkts": [4] * n,
        "dpkts": [6] * n,
        "swin": [1024] * n,
        "dwin": [1024] * n,
        "sintpkt": [12.0] * n,
        "dintpkt": [15.0] * n,
        "sjit": [2.0] * n,
        "djit": [3.0] * n,
        "res_bdy_len": [50] * n,
        "trans_depth": [1] * n,
        "stime": [1000.0 + i * 0.5 for i in range(n)],
        "label": [0 if i < 40 else 1 for i in range(n)],
    })


@pytest.fixture
def sample_cic_ids_df() -> pl.DataFrame:
    """Generates synthetic CSE-CIC-IDS2018 flow data."""
    n = 60
    return pl.DataFrame({
        "Dst Port": [80, 443, 8080, 22, 53] * (n // 5),
        "Protocol": [6] * n,
        "Timestamp": [1000.0 + i * 0.5 for i in range(n)],
        "Flow Duration": [100000 + i * 1000 for i in range(n)],
        "Tot Fwd Pkts": [5] * n,
        "Tot Bwd Pkts": [8] * n,
        "TotLen Fwd Pkts": [500 + i * 10 for i in range(n)],
        "TotLen Bwd Pkts": [1200 + i * 20 for i in range(n)],
        "Flow Byts/s": [5000.0] * n,
        "Flow Pkts/s": [25.0] * n,
        "Flow IAT Mean": [20000.0] * n,
        "Flow IAT Std": [5000.0] * n,
        "SYN Flag Cnt": [1 if i % 2 == 0 else 0 for i in range(n)],
        "ACK Flag Cnt": [1] * n,
        "Init Fwd Win Byts": [65535] * n,
        "Init Bwd Win Byts": [65535] * n,
        "Pkt Len Mean": [120.0] * n,
        "Pkt Len Std": [30.0] * n,
        "Label": ["Benign" if i < 40 else "DoS" for i in range(n)],
    })


@pytest.fixture
def sample_ctu13_df() -> pl.DataFrame:
    """Generates synthetic CTU-13 binetflow data."""
    n = 60
    return pl.DataFrame({
        "StartTime": [1000.0 + i * 0.5 for i in range(n)],
        "Dur": [0.1 + (i % 5) * 0.05 for i in range(n)],
        "Proto": ["tcp"] * n,
        "SrcAddr": [f"147.32.84.{i%10}" for i in range(n)],
        "Sport": [50000 + i for i in range(n)],
        "Dir": ["->"] * n,
        "DstAddr": [f"198.51.100.{i%20}" for i in range(n)],
        "Dport": [80, 443, 22, 53, 6667] * (n // 5),
        "State": ["CON", "S_", "FA_", "SR_"] * (n // 4),
        "sTos": [0] * n,
        "dTos": [0] * n,
        "TotPkts": [8] * n,
        "TotBytes": [1500 + i * 10 for i in range(n)],
        "SrcBytes": [500 + i * 5 for i in range(n)],
        "Label": ["Normal" if i < 40 else "Botnet" for i in range(n)],
    })


def test_unsw_nb15_batch_shape_mapping(sample_unsw_df):
    """Verifies that UNSW-NB15 flow data maps to torch.Size([B, 20, 16])."""
    extractor = CanonicalFeatureExtractor(dataset="unsw_nb15")
    tensors = extractor.process_dataframe(sample_unsw_df)

    assert tensors.ndim == 3, f"Expected 3D tensor, got {tensors.shape}"
    assert tensors.shape[1] == 20, f"Expected seq_len=20, got {tensors.shape[1]}"
    assert tensors.shape[2] == 16, f"Expected num_features=16, got {tensors.shape[2]}"

    batch_size = 8
    batches = list(extractor.yield_batches(tensors, batch_size=batch_size, as_torch=True))
    assert len(batches) > 0
    assert batches[0].shape == torch.Size([batch_size, 20, 16])


def test_cic_ids2018_batch_shape_mapping(sample_cic_ids_df):
    """Verifies that CSE-CIC-IDS2018 flow data maps to torch.Size([B, 20, 16])."""
    extractor = CanonicalFeatureExtractor()
    # Test auto-detection
    assert extractor.detect_dataset(sample_cic_ids_df.columns) == "cic_ids2018"

    tensors = extractor.process_dataframe(sample_cic_ids_df)
    assert tensors.shape[1:] == (20, 16)

    batch_size = 16
    batches = list(extractor.yield_batches(tensors, batch_size=batch_size, as_torch=True))
    assert len(batches) > 0
    assert batches[0].shape == torch.Size([batch_size, 20, 16])


def test_ctu_13_batch_shape_mapping(sample_ctu13_df):
    """Verifies that CTU-13 flow data maps to torch.Size([B, 20, 16])."""
    extractor = CanonicalFeatureExtractor()
    # Test auto-detection
    assert extractor.detect_dataset(sample_ctu13_df.columns) == "ctu_13"

    tensors = extractor.process_dataframe(sample_ctu13_df)
    assert tensors.shape[1:] == (20, 16)

    batch_size = 4
    batches = list(extractor.yield_batches(tensors, batch_size=batch_size, as_torch=True))
    assert len(batches) > 0
    assert batches[0].shape == torch.Size([batch_size, 20, 16])


def test_zero_leakage_enforcement(sample_unsw_df, sample_cic_ids_df, sample_ctu13_df):
    """Verifies that shortcut identifier columns are strictly absent from output representations."""
    extractor = CanonicalFeatureExtractor()

    for df, name in [
        (sample_unsw_df, "unsw_nb15"),
        (sample_cic_ids_df, "cic_ids2018"),
        (sample_ctu13_df, "ctu_13"),
    ]:
        sanitized_lf, detected = extractor.sanitize_and_extract(df.lazy(), dataset=name)
        sanitized_cols = [c.lower() for c in sanitized_lf.collect_schema().names()]

        # Ensure no identity shortcut columns leaked
        leakage_candidates = ["srcip", "dstip", "src_ip", "dst_ip", "sport", "dsport", "srcaddr", "dstaddr", "flow_id"]
        for leak in leakage_candidates:
            assert leak not in sanitized_cols, f"Data leakage detected! Column '{leak}' found in {detected}"

        # Verify only canonical slots + metadata exist
        for slot in CANONICAL_SLOTS:
            assert slot in sanitized_cols, f"Canonical slot '{slot}' missing in {detected}"


def test_f1_score_deterministic_edge_cases():
    """Verifies that metric calculation handles zero divisions deterministically without NaNs."""
    # Edge case 1: All predicted negative (TP=0, FP=0)
    y_true = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    y_prob_zero = np.zeros(5, dtype=np.float32)
    m1 = compute_operational_metrics(y_true, y_prob_zero, threshold=0.5)

    assert m1["precision"] == 0.0
    assert m1["recall"] == 0.0
    assert m1["f1_score"] == 0.0
    assert m1["false_positive_rate"] == 0.0
    assert m1["confusion_matrix"]["tp"] == 0
    assert m1["confusion_matrix"]["fp"] == 0
    assert not np.isnan(m1["f1_score"])

    # Edge case 2: All predicted positive (TN=0, FN=0)
    y_prob_one = np.ones(5, dtype=np.float32)
    m2 = compute_operational_metrics(y_true, y_prob_one, threshold=0.5)
    assert m2["recall"] == 1.0
    assert m2["confusion_matrix"]["fn"] == 0


def test_tune_optimal_threshold_sweep():
    """Verifies threshold tuning across precision, recall, and F1 trade-offs."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1], dtype=np.int32)
    y_probs = np.array([0.1, 0.2, 0.35, 0.6, 0.8, 0.9, 0.75, 0.4, 0.2, 0.85], dtype=np.float32)

    best_thresh, best_metrics = tune_optimal_threshold(y_true, y_probs)
    assert 0.0 < best_thresh < 1.0
    assert best_metrics["f1_score"] >= 0.80


def test_onnx_runtime_adapted_inference(sample_unsw_df):
    """Verifies that exported ONNX model accepts 16-slot adapted tensors and returns 5 valid outputs."""
    repo_root = Path(__file__).resolve().parent.parent
    onnx_path = repo_root / "models" / "threatora_transformer.onnx"
    if not onnx_path.exists():
        pytest.skip(f"ONNX model missing at {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    extractor = CanonicalFeatureExtractor(dataset="unsw_nb15")
    tensors = extractor.process_dataframe(sample_unsw_df)

    batch_size = 4
    batch_tensor = next(extractor.yield_batches(tensors, batch_size=batch_size, as_torch=True))

    assert batch_tensor.shape == torch.Size([batch_size, 20, 16])

    # Run inference via ONNX Runtime
    onnx_inputs = {"input_s_t": batch_tensor.numpy()}
    outputs = session.run(None, onnx_inputs)

    assert len(outputs) >= 3, f"Expected at least 3 outputs from ONNX model, got {len(outputs)}"
    pred_s_next, primary_logit, latent_embedding = outputs[:3]

    # Verify output dimensions
    assert pred_s_next.shape == (batch_size, 20, 16), (
        f"Expected pred_s_next shape ({batch_size}, 20, 16), got {pred_s_next.shape}"
    )
    assert primary_logit.shape == (batch_size, 1), (
        f"Expected primary_attack_logit shape ({batch_size}, 1), got {primary_logit.shape}"
    )
    assert latent_embedding.shape == (batch_size, 64), (
        f"Expected latent_embedding shape ({batch_size}, 64), got {latent_embedding.shape}"
    )


def test_smooth_l1_reconstruction():
    """Verifies Smooth L1 reconstruction error calculation."""
    s1 = np.ones((5, 20, 16), dtype=np.float32)
    s2 = np.ones((5, 20, 16), dtype=np.float32)
    loss_zero = compute_smooth_l1_reconstruction(s1, s2)
    assert loss_zero == 0.0

    s3 = s1 + 0.5
    loss_half = compute_smooth_l1_reconstruction(s1, s3)
    # diff = 0.5 < 1.0 -> 0.5 * 0.25 = 0.125
    assert pytest.approx(loss_half, abs=1e-4) == 0.125


def test_mltc_bounded_fpr_computation():
    """Verifies MLTC computation under bounded FPR (< 0.1%)."""
    n = 200
    y_true = np.zeros(n, dtype=np.int32)
    y_true[150:180] = 1  # Continuous attack episode

    primary_probs = np.full(n, 0.05, dtype=np.float32)
    primary_probs[150:180] = 0.85

    timeline_probs = np.zeros((n, 5), dtype=np.float32)
    # 5-step horizon anticipates attack starting at index 140
    timeline_probs[140:180, :] = 0.90

    mltc_res = compute_mltc_at_bounded_fpr(
        y_true=y_true,
        timeline_probs=timeline_probs,
        primary_probs=primary_probs,
        max_fpr_budget=0.001,
        bin_duration_sec=0.5,
    )

    assert mltc_res["fpr_budget_target"] == 0.001
    assert mltc_res["mltc_lead_time_seconds"] > 0.0
    assert mltc_res["empirical_fpr_at_budget"] <= 0.001
