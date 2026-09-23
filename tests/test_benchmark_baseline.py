"""Unit tests for comparative baseline ablation benchmarking (src/benchmark_baseline.py)."""

import json
import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from src.benchmark_baseline import (
    format_markdown_table,
    load_dataset,
)


def test_load_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "test_data.parquet"
        df = pl.DataFrame({
            "s_t": [np.random.randn(240).tolist(), np.random.randn(240).tolist()],
            "s_next": [np.random.randn(240).tolist(), np.random.randn(240).tolist()],
            "label": [0, 1],
        })
        df.write_parquet(p)

        X, y = load_dataset(p)
        assert X.shape == (2, 240)
        assert y.shape == (2,)
        assert y.tolist() == [0, 1]


def test_format_markdown_table():
    baseline = {
        "input_paradigm": "Static 1D vector",
        "lead_time": "0s",
        "prediction_horizon": "Instantaneous",
        "train_roc_auc": 0.90,
        "train_pr_auc": 0.85,
        "false_positive_rate": 0.05,
        "cpu_latency_ms": 0.1,
        "disk_footprint_mb": 0.01,
    }
    flagship = {
        "input_paradigm": "Temporal Sequence",
        "lead_time": "47s",
        "prediction_horizon": "5-step horizon",
        "train_roc_auc": 0.99,
        "train_pr_auc": 0.98,
        "false_positive_rate": 0.01,
        "cpu_latency_ms": 0.3,
        "disk_footprint_mb": 0.35,
    }
    table = format_markdown_table(baseline, flagship)
    assert "Comparative Ablation" in table
    assert "Logistic Regression" in table
    assert "Threatora Temporal Transformer" in table
