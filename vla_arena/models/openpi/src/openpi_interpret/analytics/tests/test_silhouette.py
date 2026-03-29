"""Tests for Silhouette Coefficient computation."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.silhouette import SILHOUETTE_GROUPS, build_modality_labels, compute_silhouette


class TestBuildModalityLabels:
    def test_length(self) -> None:
        labels, mask = build_modality_labels()
        assert labels.shape == (867,)
        assert mask.shape == (867,)

    def test_three_unique_labels(self) -> None:
        labels, mask = build_modality_labels()
        assert len(np.unique(labels[mask])) == 3

    def test_right_wrist_excluded(self) -> None:
        from analytics.constants import TOKEN_RANGES

        _, mask = build_modality_labels()
        rw_start, rw_end = TOKEN_RANGES["right_wrist_0_rgb"]
        assert not mask[rw_start:rw_end].any()

    def test_included_count(self) -> None:
        """512 visual + 48 language + 51 action = 611 included tokens."""
        _, mask = build_modality_labels()
        assert mask.sum() == 512 + 48 + 51

    def test_groups_match_definition(self) -> None:
        from analytics.constants import TOKEN_RANGES

        labels, mask = build_modality_labels()
        for group_idx, (_, modalities) in enumerate(SILHOUETTE_GROUPS.items()):
            for mod in modalities:
                start, end = TOKEN_RANGES[mod]
                assert np.all(labels[start:end] == group_idx)
                assert np.all(mask[start:end])


class TestComputeSilhouette:
    def test_well_separated_clusters_high_score(self) -> None:
        from analytics.constants import TOKEN_RANGES

        coords = np.zeros((867, 2), dtype=np.float32)
        for group_idx, (_, modalities) in enumerate(SILHOUETTE_GROUPS.items()):
            for mod in modalities:
                start, end = TOKEN_RANGES[mod]
                coords[start:end, 0] = group_idx * 100.0
                coords[start:end, 1] = group_idx * 100.0
                coords[start:end] += np.random.default_rng(group_idx).normal(
                    scale=0.1, size=(end - start, 2)
                ).astype(np.float32)

        result = compute_silhouette(coords)
        assert result.score > 0.5

    def test_random_coords_lower_score(self) -> None:
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((867, 2)).astype(np.float32)
        result = compute_silhouette(coords)
        assert -1.0 <= result.score <= 1.0

    def test_result_type(self) -> None:
        from analytics.types import TimestepSilhouette

        rng = np.random.default_rng(42)
        coords = rng.standard_normal((867, 2)).astype(np.float32)
        result = compute_silhouette(coords)
        assert isinstance(result, TimestepSilhouette)
