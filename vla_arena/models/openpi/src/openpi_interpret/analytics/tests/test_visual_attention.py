"""Tests for visual attention attribution ranking."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.visual_attention import (
    UNIFORM_VISUAL_BASELINE,
    compute_visual_attention,
    rank_layers_and_heads,
)


class TestComputeVisualAttention:
    def test_uniform_attention_gives_baseline(self) -> None:
        attention = np.ones((8, 51, 867), dtype=np.float32) / 867.0
        result = compute_visual_attention(attention)
        assert result.shape == (8,)
        for h in range(8):
            assert abs(result[h] - UNIFORM_VISUAL_BASELINE) < 1e-4

    def test_all_visual_attention_gives_one(self) -> None:
        attention = np.zeros((8, 51, 867), dtype=np.float32)
        attention[:, :, :768] = 1.0 / 768.0
        result = compute_visual_attention(attention)
        for h in range(8):
            assert abs(result[h] - 1.0) < 1e-5

    def test_zero_visual_attention_gives_zero(self) -> None:
        attention = np.zeros((8, 51, 867), dtype=np.float32)
        attention[:, :, 768:] = 1.0 / 99.0
        result = compute_visual_attention(attention)
        for h in range(8):
            assert abs(result[h]) < 1e-5

    def test_skips_state_query(self) -> None:
        """State is suffix index 0; only action queries (1-50) are used."""
        attention = np.zeros((8, 51, 867), dtype=np.float32)
        attention[:, 0, :768] = 1.0 / 768.0
        attention[:, 1:, 768:] = 1.0 / 99.0
        result = compute_visual_attention(attention)
        for h in range(8):
            assert abs(result[h]) < 1e-5

    def test_per_head_variation(self) -> None:
        attention = np.zeros((8, 51, 867), dtype=np.float32)
        attention[0, 1:, :768] = 1.0 / 768.0
        attention[1, 1:, 768:] = 1.0 / 99.0
        result = compute_visual_attention(attention)
        assert result[0] > 0.99
        assert result[1] < 0.01


class TestRankLayersAndHeads:
    def test_descending_layer_order(self) -> None:
        data = {
            0: np.array([0.5] * 8),
            3: np.array([0.9] * 8),
            6: np.array([0.7] * 8),
        }
        ranked = rank_layers_and_heads(data)
        assert ranked[0]["layer"] == 3
        assert ranked[1]["layer"] == 6
        assert ranked[2]["layer"] == 0

    def test_heads_sorted_descending_with_indices(self) -> None:
        data = {0: np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4])}
        ranked = rank_layers_and_heads(data)
        heads = ranked[0]["heads"]
        shares = [h["visual_share"] for h in heads]
        assert shares == sorted(shares, reverse=True)
        assert heads[0]["head"] == 1
        assert heads[0]["visual_share"] == pytest.approx(0.9)
        assert heads[-1]["head"] == 0
        assert heads[-1]["visual_share"] == pytest.approx(0.1)

    def test_layer_mean_is_correct(self) -> None:
        scores = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9])
        data = {17: scores}
        ranked = rank_layers_and_heads(data)
        assert abs(ranked[0]["layer_mean"] - float(scores.mean())) < 1e-6

    def test_all_head_indices_present(self) -> None:
        data = {0: np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])}
        ranked = rank_layers_and_heads(data)
        indices = {h["head"] for h in ranked[0]["heads"]}
        assert indices == {0, 1, 2, 3, 4, 5, 6, 7}
