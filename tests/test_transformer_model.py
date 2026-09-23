"""Unit tests for ThreatoraTemporalTransformerWorldModel (src/model.py)."""

import pytest
import torch
from src.model import ThreatoraTemporalTransformerWorldModel


def test_model_forward_shapes():
    batch_size = 8
    seq_len = 20
    num_features = 16

    model = ThreatoraTemporalTransformerWorldModel(
        seq_len=seq_len,
        num_features=num_features,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_layers=2,
        dropout=0.1,
        horizon_k=5,
    )
    model.eval()

    dummy_input = torch.randn(batch_size, seq_len, num_features)
    pred_s_next, primary_attack_logit, latent_embedding, mitre_logits, attn_weights = model(dummy_input)

    # Validate output tensor shapes
    assert pred_s_next.shape == (batch_size, seq_len, num_features), (
        f"Expected pred_s_next shape ({batch_size}, {seq_len}, {num_features}), got {pred_s_next.shape}"
    )
    assert primary_attack_logit.shape == (batch_size, 1), (
        f"Expected primary_attack_logit shape ({batch_size}, 1), got {primary_attack_logit.shape}"
    )
    assert latent_embedding.shape == (batch_size, 64), (
        f"Expected latent_embedding shape ({batch_size}, 64), got {latent_embedding.shape}"
    )
    assert mitre_logits.shape == (batch_size, 6), (
        f"Expected mitre_logits shape ({batch_size}, 6), got {mitre_logits.shape}"
    )
    assert attn_weights.shape == (batch_size, 4, seq_len, seq_len), (
        f"Expected attn_weights shape ({batch_size}, 4, {seq_len}, {seq_len}), got {attn_weights.shape}"
    )


def test_predict_timeline():
    batch_size = 4
    seq_len = 20
    num_features = 16
    horizon_k = 5

    model = ThreatoraTemporalTransformerWorldModel(
        seq_len=seq_len,
        num_features=num_features,
        d_model=64,
        horizon_k=horizon_k,
    )
    model.eval()

    dummy_input = torch.randn(batch_size, seq_len, num_features)
    timeline_probs = model.predict_timeline(dummy_input)

    assert timeline_probs.shape == (batch_size, horizon_k), (
        f"Expected timeline_probs shape ({batch_size}, {horizon_k}), got {timeline_probs.shape}"
    )
    # Check probabilities are in [0, 1]
    assert torch.all(timeline_probs >= 0.0) and torch.all(timeline_probs <= 1.0), (
        "Predicted timeline probabilities must be within [0, 1]"
    )


def test_learnable_positional_embeddings():
    model = ThreatoraTemporalTransformerWorldModel(num_features=16)
    assert model.pos_embedding.shape == (1, 20, 64)
    assert model.pos_embedding.requires_grad is True


def test_mitre_head_classification():
    batch_size = 3
    model = ThreatoraTemporalTransformerWorldModel(num_features=16)
    model.eval()

    dummy_input = torch.randn(batch_size, 20, 16)
    _, _, _, mitre_logits, _ = model(dummy_input)

    assert mitre_logits.shape == (batch_size, 6)
    pred_stages = torch.argmax(mitre_logits, dim=-1)
    assert pred_stages.shape == (batch_size,)
    for s in pred_stages:
        assert 0 <= s.item() <= 5

