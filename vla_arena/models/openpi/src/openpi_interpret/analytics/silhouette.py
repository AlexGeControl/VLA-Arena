"""Silhouette Coefficient on t-SNE projections.

Measures cluster separation quality using modality group labels on the
2-D t-SNE coordinates from the last sampled Transformer layer.

Cluster definition (3 groups):
  - Visual:   base_0_rgb + left_wrist_0_rgb (right wrist excluded — zero placeholder)
  - Language:  language tokens
  - Action:    state + action tokens (both Expert 1)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score

from analytics.constants import TOKEN_RANGES
from analytics.types import TimestepSilhouette

SILHOUETTE_GROUPS: dict[str, list[str]] = {
    "visual": ["base_0_rgb", "left_wrist_0_rgb"],
    "language": ["language"],
    "action": ["state", "action"],
}

_LABELS: np.ndarray | None = None
_MASK: np.ndarray | None = None


def build_modality_labels() -> tuple[np.ndarray, np.ndarray]:
    """Build cluster labels and an inclusion mask for 867 tokens.

    Right wrist tokens (512-767) are excluded via the mask since
    that camera is a zero placeholder for Pi-Zero on VLA-Arena.

    Returns:
        Tuple of (labels [867], mask [867] bool). Only tokens where
        mask is True should be passed to silhouette_score.
    """
    global _LABELS, _MASK  # noqa: PLW0603
    if _LABELS is not None and _MASK is not None:
        return _LABELS, _MASK

    labels = np.empty(867, dtype=np.int32)
    mask = np.zeros(867, dtype=bool)

    for group_idx, (_, modalities) in enumerate(SILHOUETTE_GROUPS.items()):
        for modality in modalities:
            start, end = TOKEN_RANGES[modality]
            labels[start:end] = group_idx
            mask[start:end] = True

    _LABELS = labels
    _MASK = mask
    return _LABELS, _MASK


def compute_silhouette(tsne_coords: np.ndarray) -> TimestepSilhouette:
    """Compute the Silhouette Coefficient on t-SNE coordinates.

    Excludes right wrist tokens (zero placeholder) and groups state
    with action tokens (both Expert 1).

    Args:
        tsne_coords: Shape ``[867, 2]`` float32 from a single
            (timestep, layer).

    Returns:
        TimestepSilhouette with the score in ``[-1, 1]``.
    """
    labels, mask = build_modality_labels()
    score = silhouette_score(
        tsne_coords[mask], labels[mask], metric="euclidean"
    )
    return TimestepSilhouette(score=float(score))
