"""Visual attention attribution ranking across layers and heads.

Computes the fraction of action query attention directed at visual tokens
(indices 0-767) for each (layer, head) pair, then ranks layers and heads
by visual attention share to quantify visual primacy in the attention routing.
"""

from __future__ import annotations

import numpy as np

from analytics.constants import SAMPLED_LAYERS, VISUAL_RANGE, NUM_HEADS

UNIFORM_VISUAL_BASELINE: float = (VISUAL_RANGE[1] - VISUAL_RANGE[0]) / 867


def compute_visual_attention(attention: np.ndarray) -> np.ndarray:
    """Per-head visual attention share for action queries.

    Args:
        attention: Shape ``[8, 51, 867]``. Rows sum to 1.0.

    Returns:
        Array of shape ``[8]`` — visual attention share per head,
        averaged across the 50 action queries (suffix indices 1-50).
    """
    action_attn = attention[:, 1:, :]
    v_start, v_end = VISUAL_RANGE
    visual_mass = action_attn[:, :, v_start:v_end].sum(axis=-1)
    return visual_mass.mean(axis=-1).astype(np.float64)


def rank_layers_and_heads(
    per_layer_scores: dict[int, np.ndarray],
) -> list[dict]:
    """Rank layers descending by mean visual share, heads within each layer.

    Args:
        per_layer_scores: Maps layer index to ``[8]`` per-head visual shares.

    Returns:
        List of dicts sorted descending by ``layer_mean``, each containing:
        ``layer``, ``layer_mean``, ``head_scores`` (sorted descending).
    """
    entries = []
    for layer, scores in per_layer_scores.items():
        layer_mean = float(scores.mean())
        indexed_heads = [
            {"head": int(i), "visual_share": float(s)}
            for i, s in enumerate(scores)
        ]
        indexed_heads.sort(key=lambda h: h["visual_share"], reverse=True)
        entries.append({
            "layer": layer,
            "layer_mean": layer_mean,
            "heads": indexed_heads,
        })
    entries.sort(key=lambda e: e["layer_mean"], reverse=True)
    return entries
