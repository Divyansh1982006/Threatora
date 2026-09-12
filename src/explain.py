"""Explainability Engine for NetForecast (NTRO PS 26153).

Answers 'Why This Prediction?' via three transparent interpretability channels:
  1. Feature Attribution: Saliency / Integrated Gradients over the 62 attributes
  2. Temporal Attention: Causal multi-head attention weights over past 16 windows
  3. Predicted State Delta: Forecasted changes in observable traffic metrics
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import torch

from .config import ALL_FEATURE_COLS
from .model.world_model import NetworkWorldModel


def compute_feature_saliency(
    model: NetworkWorldModel,
    input_tensor: torch.Tensor,
    target_head: str = "infiltration"
) -> Dict[str, float]:
    """Computes gradient-based feature attribution across the 62 input attributes.

    Args:
        model: NetworkWorldModel instance
        input_tensor: (1, seq_len, 62) PyTorch tensor
        target_head: 'infiltration' or 'stage'
    Returns:
        dict of feature name to normalized attribution percentage (0..100)
    """
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)

    out = model(x)
    if target_head == "infiltration":
        score = out["infiltration_prob"][:, -1].sum()
    else:
        # Maximum non-benign stage logit
        logits = out["stage_logits"][:, -1, 1:]
        score = logits.max()

    score.backward()

    # Average absolute gradients across sequence and normalize
    grads = x.grad.abs().squeeze(0).mean(dim=0).cpu().numpy()
    total = np.sum(grads)

    if total > 1e-6:
        norm_scores = (grads / total) * 100.0
    else:
        norm_scores = np.ones(len(ALL_FEATURE_COLS)) / len(ALL_FEATURE_COLS) * 100.0

    attributions = {
        name: float(round(norm_scores[i], 2))
        for i, name in enumerate(ALL_FEATURE_COLS)
    }

    # Sort descending by influence
    return dict(sorted(attributions.items(), key=lambda kv: kv[1], reverse=True))


def compute_state_deltas(
    current_obs: np.ndarray,
    forecasted_obs: np.ndarray,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Calculates top forecasted metric shifts between current state and future horizon."""
    curr = np.array(current_obs).flatten()
    fore = np.array(forecasted_obs).flatten()

    deltas = []
    for i, col in enumerate(ALL_FEATURE_COLS[:len(curr)]):
        diff = float(fore[i] - curr[i])
        pct_change = float((diff / max(abs(curr[i]), 1e-4)) * 100.0)
        deltas.append({
            "feature": col,
            "current_value": round(float(curr[i]), 3),
            "forecast_value": round(float(fore[i]), 3),
            "delta": round(diff, 3),
            "pct_change": round(pct_change, 1)
        })

    # Sort by absolute delta
    deltas.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return deltas[:top_k]


def generate_full_explanation(
    model: NetworkWorldModel,
    input_seq: np.ndarray,
    forecast_seq: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Generates complete explainability package for API and dashboard."""
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    tensor_in = torch.tensor(input_seq, dtype=torch.float32, device=device)
    if tensor_in.ndim == 2:
        tensor_in = tensor_in.unsqueeze(0)

    # 1. Feature Saliency
    attributions = compute_feature_saliency(model, tensor_in)
    top_5 = dict(list(attributions.items())[:5])

    # 2. Attention Weights over 16 windows
    with torch.no_grad():
        out = model(tensor_in)
        attn_weights = out["attention_weights"].squeeze(0).cpu().numpy()  # (T, T)
        last_step_attention = attn_weights[-1, :].tolist()

    # 3. State Deltas
    deltas = []
    if forecast_seq is not None and len(forecast_seq) > 0:
        current_obs = input_seq[-1]
        future_obs = forecast_seq[-1]
        deltas = compute_state_deltas(current_obs, future_obs)

    return {
        "top_features": top_5,
        "all_attributions": attributions,
        "temporal_attention_weights": [round(w, 4) for w in last_step_attention],
        "state_deltas": deltas,
        "primary_threat_driver": list(top_5.keys())[0] if top_5 else "n_flows"
    }
