"""Cross-Modal Fusion (CMF) score computation.

Computes attention-weighted cross-modal cosine similarity per head in the
shared 256-d attention head space, then aggregates across heads and queries.

The 5 CMF pairs:
  S → L  (State → Language)
  S → V  (State → Visual)
  A → L  (Action → Language)
  A → V  (Action → Visual)
  A → S  (Action → State, direct cosine — degenerate single-target case)
"""

from __future__ import annotations

import numpy as np

from analytics.constants import NUM_HEADS
from analytics.types import TimestepCmf

_EPSILON = 1e-10


def compute_all_cmf(
    q_suffix: np.ndarray,
    attended_language: np.ndarray,
    attended_visual: np.ndarray,
) -> TimestepCmf:
    """Compute all 5 CMF scores for a single timestep.

    Args:
        q_suffix: Suffix Q-projections, shape ``[51, 8, 256]``.
        attended_language: Pre-computed attended language, ``[51, 8, 256]``.
        attended_visual: Pre-computed attended visual, ``[51, 8, 256]``.

    Returns:
        TimestepCmf with all 5 pair scores.
    """
    return TimestepCmf(
        S_to_L=_cmf_from_attended(q_suffix, attended_language, [0]),
        S_to_V=_cmf_from_attended(q_suffix, attended_visual, [0]),
        A_to_L=_cmf_from_attended(q_suffix, attended_language, list(range(1, 51))),
        A_to_V=_cmf_from_attended(q_suffix, attended_visual, list(range(1, 51))),
        A_to_S=_cmf_direct(q_suffix, query_indices=list(range(1, 51)), target_index=0),
    )


def _cmf_from_attended(
    q_suffix: np.ndarray,
    attended: np.ndarray,
    query_suffix_indices: list[int],
) -> float:
    """CMF using pre-computed attended representations.

    For each head h, for each query index k:
      CMF_h(k) = cosine(q_suffix[k, h, :], attended[k, h, :])

    Aggregation: mean over heads, then mean over queries.

    Args:
        q_suffix: Shape ``[51, 8, 256]``.
        attended: Shape ``[51, 8, 256]``.
        query_suffix_indices: Suffix indices for query tokens.
    """
    q = q_suffix[query_suffix_indices]
    v = attended[query_suffix_indices]
    return _batched_cosine_mean(q, v)


def _cmf_direct(
    q_suffix: np.ndarray,
    query_indices: list[int],
    target_index: int,
) -> float:
    """CMF for single-target case (A → S).

    With 1 target token, intra-modality normalized attention is always 1,
    so the attended representation equals the target embedding. CMF reduces
    to direct cosine similarity between query and target Q-projections.

    Args:
        q_suffix: Shape ``[51, 8, 256]``.
        query_indices: Suffix indices for query tokens (actions).
        target_index: Suffix index for the single target (state).
    """
    q = q_suffix[query_indices]
    t = q_suffix[target_index:target_index + 1].repeat(len(query_indices), axis=0)
    return _batched_cosine_mean(q, t)


def _batched_cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity across queries and heads.

    Args:
        a: Shape ``[num_queries, num_heads, head_dim]``.
        b: Shape ``[num_queries, num_heads, head_dim]``.

    Returns:
        Scalar mean cosine similarity.
    """
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True).clip(min=_EPSILON)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True).clip(min=_EPSILON)
    cos = (a * b).sum(axis=-1) / (a_norm * b_norm).squeeze(-1)
    return float(cos.mean())
