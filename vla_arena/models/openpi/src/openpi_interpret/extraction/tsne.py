"""Pre-compute t-SNE projections and nearest neighbors from Q-projections.

For each sampled layer:
  1. Flatten Q-projections across heads → ``[num_tokens, num_heads * head_dim]``.
  2. Concatenate prefix + suffix → ``[total_tokens, embed_dim]``.
  3. Run ``TSNE(n_components=2)`` to get 2-D coordinates.
  4. For each action token, find the nearest neighbor in each modality group
     using a KD-tree.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import KDTree
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

MODALITY_RANGES: dict[str, tuple[int, int]] = {
    "base_0_rgb": (0, 256),
    "left_wrist_0_rgb": (256, 512),
    "right_wrist_0_rgb": (512, 768),
    "language": (768, 816),
    "state": (816, 817),
}

MODALITY_ORDER: list[str] = [
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "language",
    "state",
]

ACTION_RANGE: tuple[int, int] = (817, 867)
NEIGHBOR_DTYPE = np.dtype([("neighbor_index", "<i4"), ("distance", "<f4")])


def compute_tsne(
    q_prefix: np.ndarray,
    q_suffix: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE on concatenated Q-projections for a single layer.

    Args:
        q_prefix: Shape ``[1, prefix_len, num_heads, head_dim]``.
        q_suffix: Shape ``[1, suffix_len, num_heads, head_dim]``.
        perplexity: t-SNE perplexity parameter.
        random_state: Seed for reproducibility.

    Returns:
        Coordinates array of shape ``[total_tokens, 2]`` (float32).
    """
    prefix_flat = _flatten_heads(q_prefix[0])
    suffix_flat = _flatten_heads(q_suffix[0])
    combined = np.concatenate([prefix_flat, suffix_flat], axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, combined.shape[0] - 1),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(combined.astype(np.float32))
    return coords.astype(np.float32)


def compute_neighbors(
    tsne_coords: np.ndarray,
    num_action_tokens: int = 50,
) -> np.ndarray:
    """Find nearest neighbor per modality for each action token.

    Args:
        tsne_coords: Shape ``[total_tokens, 2]`` from ``compute_tsne``.
        num_action_tokens: Number of action tokens (default 50).

    Returns:
        Structured array of shape ``[num_action_tokens, 5]`` with dtype
        ``[("neighbor_index", int32), ("distance", float32)]``.
    """
    result = np.empty((num_action_tokens, len(MODALITY_ORDER)), dtype=NEIGHBOR_DTYPE)
    action_start, action_end = ACTION_RANGE
    action_coords = tsne_coords[action_start:action_end]

    for mod_idx, modality in enumerate(MODALITY_ORDER):
        start, end = MODALITY_RANGES[modality]
        mod_coords = tsne_coords[start:end]
        tree = KDTree(mod_coords)
        distances, indices = tree.query(action_coords, k=1)
        global_indices = (np.asarray(indices).ravel() + start).astype(np.int32)
        result[:, mod_idx]["neighbor_index"] = global_indices
        result[:, mod_idx]["distance"] = np.asarray(distances).ravel().astype(np.float32)

    return result


def compute_layer_tsne_and_neighbors(
    q_prefix: np.ndarray,
    q_suffix: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: compute both t-SNE and neighbors for one layer.

    Args:
        q_prefix: Shape ``[1, prefix_len, num_heads, head_dim]``.
        q_suffix: Shape ``[1, suffix_len, num_heads, head_dim]``.
        perplexity: t-SNE perplexity parameter.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (tsne_coords ``[total, 2]``, neighbors ``[50, 5]``).
    """
    coords = compute_tsne(q_prefix, q_suffix, perplexity, random_state)
    neighbors = compute_neighbors(coords)
    return coords, neighbors


def _flatten_heads(q: np.ndarray) -> np.ndarray:
    """Reshape ``[seq_len, num_heads, head_dim]`` → ``[seq_len, num_heads * head_dim]``."""
    seq_len, num_heads, head_dim = q.shape
    return q.reshape(seq_len, num_heads * head_dim)
