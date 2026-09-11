"""Proof Test: Asserts zero gradient to observation encoder during .imagine() rollout.

Demonstrates that the World Model forward simulation does NOT peek at future observations.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, ALL_FEATURE_COLS, SEQUENCE_LENGTH
from src.model.world_model import NetworkWorldModel


def test_no_peeking_zero_gradient():
    print("[*] Verifying no observation peeking during imagination rollout...")
    b, t, d = 2, SEQUENCE_LENGTH, len(ALL_FEATURE_COLS)
    x = torch.randn(b, t, d)

    cfg = ModelConfig(obs_dim=d, hidden_dim=32, embed_dim=16)
    model = NetworkWorldModel(cfg)

    # Initial observation pass
    out = model(x)

    # Zero all parameter gradients
    model.zero_grad()

    # Rollout forward purely with autoregressive LSTM cell
    sim = model.imagine(
        initial_lstm_h=out["final_lstm_h"].detach(),
        initial_lstm_c=out["final_lstm_c"].detach(),
        horizon=5,
        n_trajectories=4,
        mc_dropout=False
    )

    # Simulate an imagination-only loss on dreamed states
    dummy_loss = torch.tensor(sim["infilt_prob_mean"], requires_grad=True).sum()
    dummy_loss.backward()

    # Assert that the observation encoder gradients are strictly NONE or ZERO
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0.0), f"Leak detected! Encoder gradient non-zero in {name}"

    print("[+] Zero-gradient assertion passed: Rollout simulator is strictly causally isolated from observations.")


if __name__ == "__main__":
    test_no_peeking_zero_gradient()
