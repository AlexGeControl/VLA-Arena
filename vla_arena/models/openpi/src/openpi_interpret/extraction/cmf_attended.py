"""Compute attention-weighted modality representations for CMF scoring.

For each (layer, head), produces per-suffix-query attended representations
for the Language and Visual modalities using intra-modality normalized
attention weights and V-projections as token embeddings.

V-projections have a single KV head (shared across all 8 query groups).
The per-head variation in the attended representation comes solely from
the different attention weight distributions per head.

The attended representations are stored in HDF5 and consumed by the
Track D analytics pipeline to compute Cross-Modal Fusion scores.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

VISUAL_RANGE: tuple[int, int] = (0, 768)
LANGUAGE_RANGE: tuple[int, int] = (768, 816)
_EPSILON = 1e-10


def compute_cmf_attended(
    attention: np.ndarray,
    v_prefix: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute intra-modality normalized attended representations using V projections.

    Args:
        attention: Per-head attention weights, shape ``[8, 51, 867]``.
        v_prefix: Prefix V-projections, shape ``[816, 256]`` (single KV head).

    Returns:
        Dict with keys ``"language"`` and ``"visual"``, each mapping to
        a float32 array of shape ``[51, 8, 256]``.
    """
    attended_lang = _attended_for_range(attention, v_prefix, LANGUAGE_RANGE)
    attended_vis = _attended_for_range(attention, v_prefix, VISUAL_RANGE)

    return {
        "language": attended_lang.transpose(1, 0, 2).astype(np.float32),
        "visual": attended_vis.transpose(1, 0, 2).astype(np.float32),
    }


def _attended_for_range(
    attention: np.ndarray,
    v_prefix: np.ndarray,
    key_range: tuple[int, int],
) -> np.ndarray:
    """Compute attended representation for a contiguous key range.

    V-projections are shared across all heads (single KV head in GQA),
    so ``embed`` has shape ``[num_tokens, 256]``. numpy broadcasts the
    matmul ``[8, 51, N] @ [N, 256] -> [8, 51, 256]`` correctly.

    Args:
        attention: Shape ``[8, 51, 867]``.
        v_prefix: Shape ``[816, 256]``.
        key_range: ``(start, end)`` into the 867-length key axis.

    Returns:
        Shape ``[8, 51, 256]`` — per head, per suffix query, head_dim.
    """
    start, end = key_range
    alpha = attention[:, :, start:end]
    alpha_sum = alpha.sum(axis=-1, keepdims=True)
    safe_sum = np.where(alpha_sum > _EPSILON, alpha_sum, np.ones_like(alpha_sum))
    alpha_norm = np.where(alpha_sum > _EPSILON, alpha / safe_sum, 0.0)

    embed = v_prefix[start:end]

    return alpha_norm @ embed
