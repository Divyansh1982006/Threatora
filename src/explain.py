"""Self-Attention Attribution & Dynamic State Saliency Engine for Threatora (NTRO PS 26153).

Replaces misleading 'SHAP' terminology with exact Neural Interpretability channels:
  1. Self-Attention Attribution: Temporal attention weights extracted from the Transformer encoder.
  2. Dynamic State Saliency: Gradient-based feature attribution across the 16 canonical slots.
  3. Forecasted State Deltas: Predicted shifts between current state S_t and future state S_{t+1}.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .adapters.dataset_adapter import CANONICAL_SLOTS
from .model.transformer import ThreatoraTemporalTransformerWorldModel


def compute_feature_saliency(
    model: ThreatoraTemporalTransformerWorldModel,
    input_tensor: torch.Tensor,
    target_head: str = "threat",
) -> Dict[str, float]:
    """Computes gradient-based feature saliency across canonical slots.

    Args:
        model: ThreatoraTemporalTransformerWorldModel instance
        input_tensor: (1, seq_len, num_features) PyTorch tensor
        target_head: 'threat' or 'mitre'
    Returns:
        dict of slot name to normalized attribution percentage (0..100)
    """
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)

    out = model(x)
    if isinstance(out, dict):
        primary_attack_logit = None
        for key in ["attack_logits", "infilt_logits", "infiltration_prob", "infilt_prob", "attack_prob", "risk_prob"]:
            if key in out and out[key] is not None:
                primary_attack_logit = out[key]
                break
        mitre_logits = out.get("stage_logits")
        if primary_attack_logit is None:
            for v in out.values():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    primary_attack_logit = v
                    break
            if primary_attack_logit is None:
                primary_attack_logit = torch.zeros(1, device=x.device, requires_grad=True)
    elif isinstance(out, (tuple, list)):
        if len(out) >= 5:
            pred_s_next, primary_attack_logit, latent_embedding, mitre_logits, attn_weights = out[:5]
        elif len(out) >= 3:
            pred_s_next, primary_attack_logit, latent_embedding = out[:3]
            mitre_logits = None
            attn_weights = None
        else:
            primary_attack_logit = out[0]
            mitre_logits = None
    else:
        primary_attack_logit = out
        mitre_logits = None

    if target_head == "threat" or mitre_logits is None:
        score = primary_attack_logit.sum()
    else:
        # Maximum non-benign stage logit
        score = mitre_logits[:, 1:].max()

    score.backward()

    # Average absolute gradients across sequence length and normalize
    grads = x.grad.abs().squeeze(0).mean(dim=0).cpu().numpy()
    total = np.sum(grads)

    slots = CANONICAL_SLOTS[: len(grads)]
    if total > 1e-6:
        norm_scores = (grads / total) * 100.0
    else:
        norm_scores = np.ones(len(slots)) / len(slots) * 100.0

    attributions = {
        name: float(round(norm_scores[i], 2))
        for i, name in enumerate(slots)
    }

    # Sort descending by influence
    return dict(sorted(attributions.items(), key=lambda kv: kv[1], reverse=True))


def compute_state_deltas(
    current_obs: np.ndarray,
    forecasted_obs: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Calculates top forecasted metric shifts between current state S_t and future S_{t+1}."""
    curr = np.array(current_obs).flatten()
    fore = np.array(forecasted_obs).flatten()
    slots = CANONICAL_SLOTS[: len(curr)]

    deltas = []
    for i, col in enumerate(slots):
        diff = float(fore[i] - curr[i])
        pct_change = float((diff / max(abs(curr[i]), 1e-4)) * 100.0)
        deltas.append({
            "feature": col,
            "current_value": round(float(curr[i]), 3),
            "forecast_value": round(float(fore[i]), 3),
            "delta": round(diff, 3),
            "pct_change": round(pct_change, 1),
        })

    # Sort by absolute delta
    deltas.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return deltas[:top_k]


def generate_full_explanation(
    model: ThreatoraTemporalTransformerWorldModel,
    input_seq: np.ndarray,
    forecast_seq: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Generates complete Self-Attention Attribution & Dynamic State Saliency package."""
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    tensor_in = torch.tensor(input_seq, dtype=torch.float32, device=device)
    if tensor_in.ndim == 2:
        tensor_in = tensor_in.unsqueeze(0)

    # 1. Feature Saliency
    attributions = compute_feature_saliency(model, tensor_in)
    top_5 = dict(list(attributions.items())[:5])

    # 2. Multi-Head Attention Weights extraction
    with torch.no_grad():
        out = model(tensor_in)
        if isinstance(out, (tuple, list)) and len(out) >= 5:
            attn_tensor = out[4]
        elif isinstance(out, dict):
            attn_tensor = out.get("attention_weights")
        else:
            attn_tensor = None

        if attn_tensor is not None:
            # Average across attention heads -> (20, 20)
            if attn_tensor.ndim == 4:
                attn_mean = attn_tensor.squeeze(0).mean(dim=0).cpu().numpy()
            elif attn_tensor.ndim == 3:
                attn_mean = attn_tensor.mean(dim=0).cpu().numpy()
            else:
                attn_mean = attn_tensor.cpu().numpy()
            last_step_attention = attn_mean[-1, :].tolist()
            full_matrix = attn_mean.tolist()
        else:
            last_step_attention = [1.0 / 20.0] * 20
            full_matrix = np.eye(20).tolist()

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
        "attention_matrix": full_matrix,
        "state_deltas": deltas,
        "primary_threat_driver": list(top_5.keys())[0] if top_5 else "packet_rate",
        "method": "Self-Attention Attribution & Dynamic State Saliency",
    }
