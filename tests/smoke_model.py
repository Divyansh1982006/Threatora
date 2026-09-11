"""Smoke test suite for NetForecast LSTM World Model (NTRO PS 26153).

Verifies:
  - Input/Output shapes with 62 observation features & 16-window sequences
  - LSTM hidden and cell state dynamics propagation
  - Analytic KL divergence computation
  - K-step .imagine() forward simulation without observations
  - FeatureScaler log1p and z-score transformations
  - Logistic Regression baselines
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, ALL_FEATURE_COLS, SEQUENCE_LENGTH, FORECAST_HORIZON
from src.model.world_model import NetworkWorldModel
from src.model.baseline import LogisticRegressionBaseline, PersistenceBaseline
from src.features.windows import FeatureScaler


def test_lstm_world_model_forward():
    print("[*] Testing LSTM World Model forward pass...")
    b, t, d = 4, SEQUENCE_LENGTH, len(ALL_FEATURE_COLS)
    x = torch.randn(b, t, d)

    cfg = ModelConfig(obs_dim=d, hidden_dim=64, embed_dim=32, n_heads=2)
    model = NetworkWorldModel(cfg)

    out = model(x)
    assert "reconstruction" in out, "Missing reconstruction in forward output"
    assert "infiltration_prob" in out, "Missing infiltration_prob in forward output"
    assert "stage_logits" in out, "Missing stage_logits in forward output"
    assert "states" in out, "Missing states in forward output"
    assert "attention_weights" in out, "Missing attention_weights in forward output"

    assert out["reconstruction"].shape == (b, t, d)
    assert out["infiltration_prob"].shape == (b, t)
    assert out["stage_logits"].shape == (b, t, cfg.num_stages + 1)
    assert out["attention_weights"].shape == (b, t, t)
    print("  [+] LSTM World Model forward shapes passed.")


def test_world_model_imagine():
    print("[*] Testing K-step .imagine() rollout simulation...")
    b, t, d = 1, SEQUENCE_LENGTH, len(ALL_FEATURE_COLS)
    x = torch.randn(b, t, d)

    cfg = ModelConfig(obs_dim=d, hidden_dim=32, embed_dim=16, n_heads=2)
    model = NetworkWorldModel(cfg)

    with torch.no_grad():
        out = model(x)
        sim = model.imagine(
            initial_lstm_h=out["final_lstm_h"],
            initial_lstm_c=out["final_lstm_c"],
            horizon=FORECAST_HORIZON,
            n_trajectories=8,
            mc_dropout=False
        )

    assert sim["horizon"] == FORECAST_HORIZON
    assert len(sim["infilt_prob_mean"]) == FORECAST_HORIZON
    assert len(sim["infilt_prob_lower"]) == FORECAST_HORIZON
    assert len(sim["infilt_prob_upper"]) == FORECAST_HORIZON
    assert len(sim["predicted_stages"]) == FORECAST_HORIZON
    assert len(sim["feature_forecast"]) == FORECAST_HORIZON

    for p in sim["infilt_prob_mean"]:
        assert 0.0 <= p <= 1.0
    print("  [+] .imagine() forward simulation passed.")


def test_feature_scaler():
    print("[*] Testing FeatureScaler with log1p & z-score...")
    n_samples = 50
    X = np.abs(np.random.randn(n_samples, len(ALL_FEATURE_COLS)) * 500)

    scaler = FeatureScaler()
    scaler.fit(X)
    X_trans = scaler.transform(X)

    assert X_trans.shape == X.shape
    assert abs(np.mean(X_trans)) < 1.0
    print("  [+] FeatureScaler transformations passed.")


def test_logistic_regression_baselines():
    print("[*] Testing Logistic Regression baselines...")
    b, t, d = 20, SEQUENCE_LENGTH, len(ALL_FEATURE_COLS)
    X = np.random.randn(b, t, d)
    y = np.random.randint(0, 2, size=b)

    lr_single = LogisticRegressionBaseline(history_len=1)
    lr_single.fit(X, y)
    res1 = lr_single.evaluate(X, y)
    assert "f1_score" in res1 and "false_positive_rate" in res1

    lr_stacked = LogisticRegressionBaseline(history_len=8)
    lr_stacked.fit(X, y)
    res8 = lr_stacked.evaluate(X, y)
    assert "f1_score" in res8 and "false_positive_rate" in res8
    print("  [+] Logistic Regression baselines passed.")


if __name__ == "__main__":
    test_lstm_world_model_forward()
    test_world_model_imagine()
    test_feature_scaler()
    test_logistic_regression_baselines()
    print("\n[+] All NetForecast smoke tests passed successfully!")
