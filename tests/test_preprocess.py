"""Unit tests for high-performance preprocessing and windowing pipeline (src/preprocess.py)."""

import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import pytest
import joblib
from sklearn.preprocessing import RobustScaler

from src.preprocess import (
    UNSWPreprocessor,
    TARGET_FEATURE_NAMES,
    SHORTCUT_IDENTIFIER_COLS,
    parse_port_value,
    run_pipeline,
)


def test_target_features_count():
    assert len(TARGET_FEATURE_NAMES) == 16
    assert TARGET_FEATURE_NAMES == [
        "duration_norm",
        "byte_ratio",
        "packet_rate",
        "iat_mean",
        "iat_std",
        "ttl_mean",
        "ttl_variance",
        "tcp_syn_ratio",
        "tcp_ack_ratio",
        "tcp_window_norm",
        "is_privileged_port",
        "payload_entropy",
        "tcp_rst_ratio",
        "fwd_bwd_packet_ratio",
        "payload_bytes_mean",
        "iat_max_norm",
    ]


def test_parse_port_value():
    assert parse_port_value(80) == 80
    assert parse_port_value("443") == 443
    assert parse_port_value("0x0050") == 80
    assert parse_port_value("0xc0a8") == 49320
    assert parse_port_value(None) == 0
    assert parse_port_value("-") == 0


def test_clean_and_sanitize_zero_leakage():
    # Synthetic lazy dataframe with shortcut columns
    df = pl.DataFrame({
        "srcip": ["10.0.0.1", "10.0.0.2"],
        "dstip": ["192.168.1.1", "192.168.1.2"],
        "sport": ["45000", "0xc0a8"],
        "dsport": ["80", "8080"],
        "stime": [1000.0, 1000.5],
        "dur": [0.1, 0.2],
        "sbytes": [100, 200],
        "dbytes": [150, 250],
        "spkts": [2, 4],
        "dpkts": [3, 5],
        "state": ["CON", "FIN"],
        "sintpkt": [10.0, 20.0],
        "dintpkt": [12.0, 22.0],
        "label": [0, 1],
    })

    prep = UNSWPreprocessor()
    sanitized = prep.clean_and_sanitize(df.lazy()).collect()

    # Verify shortcut columns are stripped
    for shortcut in ["srcip", "dstip", "sport", "dsport", "flow_id"]:
        assert shortcut not in sanitized.columns

    # Verify is_privileged_port
    # Row 0: port 80 -> privileged (1)
    # Row 1: port 8080 -> not privileged (0)
    assert sanitized["is_privileged_port"].to_list() == [1, 0]
    assert sanitized["label"].to_list() == [0, 1]


def test_windowing_and_parquet_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        prep = UNSWPreprocessor(
            bin_duration_sec=0.5,
            sequence_length=20,
            stride=1,
            train_ratio=0.8,
            output_dir=tmp_path,
        )

        # 100 bins with 16 features
        n_bins = 100
        features = np.random.randn(n_bins, 16).astype(np.float32)
        labels = np.zeros(n_bins, dtype=np.int32)
        mitre = np.zeros(n_bins, dtype=np.int32)
        labels[25] = 1  # Malicious bin in train split
        mitre[25] = 2

        # Split
        train_f, train_l, train_m, val_f, val_l, val_m = prep.chronological_split(features, labels, mitre)
        assert len(train_f) == 80
        assert len(val_f) == 20

        # Scale
        train_s, val_s, scaler = prep.fit_and_scale(train_f, val_f)
        assert (tmp_path / "scaler.joblib").exists()

        loaded_scaler = joblib.load(tmp_path / "scaler.joblib")
        assert isinstance(loaded_scaler, RobustScaler)

        # Window generation: span = 40 (20 + 20)
        # Train has 80 bins: (80 - 40) // 1 + 1 = 41 windows
        s_t, s_next, y, y_hor, mitre_w = prep.generate_paired_windows(train_s, train_l, train_m)
        assert s_t.shape == (41, 20, 16)
        assert s_next.shape == (41, 20, 16)
        assert len(y) == 41
        assert y_hor.shape == (41, 5)
        assert len(mitre_w) == 41
        # Bin 25 is malicious, so windows covering bin 25 will have label 1
        assert np.sum(y) > 0

        # Parquet save & read back
        out_parquet = tmp_path / "train_windows.parquet"
        prep.save_parquet(s_t, s_next, y, y_hor, mitre_w, out_parquet)
        assert out_parquet.exists()

        read_df = pl.read_parquet(out_parquet)
        assert len(read_df) == 41
        assert "s_t" in read_df.columns
        assert "s_next" in read_df.columns
        assert "label" in read_df.columns
        assert "y_horizon" in read_df.columns
        assert "mitre_stage" in read_df.columns
        assert len(read_df["s_t"][0]) == 320
        assert len(read_df["s_next"][0]) == 320

