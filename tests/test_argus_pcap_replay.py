"""Unit tests for targeted Argus+PCAP ingestion and Experience Replay Buffer."""

import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import pytest
import torch
import torch.nn as nn

from src.ingest_argus_pcap import (
    parse_argus_csv_file,
    build_experience_replay_buffer,
    generate_paired_windows,
)
from src.adapters.dataset_adapter import CanonicalFeatureExtractor, CANONICAL_SLOTS
from src.model.transformer import ThreatoraTemporalTransformerWorldModel


def test_argus_csv_parsing_zero_leakage():
    """Verify that parsing Argus CSVs enforces zero data leakage and extracts canonical slots."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
        # Synthetic Argus line: 33 columns
        tmp.write("1421927414,tcp,175.45.176.0,13284,149.171.126.16,80,20,FIN,1421927416,14,6,1362,268,7.948,254,252,4233.6,749.6,6,1,183.5,474.2,18786.7,941.7,255,3897219059,2466816006,255,0.066,0.017,0.048,97,45\n")
        tmp_path = Path(tmp.name)

    try:
        extractor = CanonicalFeatureExtractor()
        df = parse_argus_csv_file(tmp_path, extractor)

        assert df is not None
        assert len(df) == 1

        # Check that all canonical continuous slots are present
        for slot in CANONICAL_SLOTS:
            assert slot in df.columns

        # Check zero data leakage: raw identifiers must be omitted
        for shortcut in ["srcip", "dstip", "sport", "dsport", "stime", "ltime"]:
            assert shortcut not in df.columns

        # Verify privileged port (destination port 80 -> 1.0)
        assert df["is_privileged_port"][0] == 1.0

        # Verify attack label for attacker IP 175.45.176.0
        assert df["label"][0] == 1
        assert df["mitre_stage"][0] == 2

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_experience_replay_buffer_sampling():
    """Verify that Experience Replay Buffer correctly samples stratified subset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "dummy_train.parquet"

        # Create dummy parquet with 100 samples (60 attack, 40 benign)
        n = 100
        data = {
            "s_t": [list(np.zeros(320, dtype=np.float32)) for _ in range(n)],
            "s_next": [list(np.zeros(320, dtype=np.float32)) for _ in range(n)],
            "label": [1] * 60 + [0] * 40,
            "y_horizon": [list(np.zeros(5, dtype=np.float32)) for _ in range(n)],
            "mitre_stage": [2] * 60 + [0] * 40,
        }
        pl.DataFrame(data).write_parquet(tmp_path)

        replay_df = build_experience_replay_buffer(tmp_path, sample_ratio=0.30, seed=42)

        # 30% of 60 is 18, 30% of 40 is 12 -> total ~30
        assert len(replay_df) == 30
        assert (replay_df["label"] == 1).sum() == 18
        assert (replay_df["label"] == 0).sum() == 12
        assert "s_t" in replay_df.columns
        assert "s_next" in replay_df.columns


def test_gradient_clipping_integration():
    """Verify that torch.nn.utils.clip_grad_norm_ operates safely on the model."""
    model = ThreatoraTemporalTransformerWorldModel(num_features=16)
    x = torch.randn(4, 20, 16)
    target_s_next = torch.randn(4, 20, 16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()

    pred_s_next, _, _, _, _ = model(x)
    loss = criterion(pred_s_next, target_s_next)
    loss.backward()

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert grad_norm is not None
    assert float(grad_norm) >= 0.0

    optimizer.step()
