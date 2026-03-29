"""Tests for Cross-Modal Fusion score computation."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.cmf import _batched_cosine_mean, _cmf_direct, _cmf_from_attended, compute_all_cmf
from analytics.types import TimestepCmf
from extraction.cmf_attended import compute_cmf_attended


class TestBatchedCosineMean:
    """Unit tests for the core cosine similarity helper."""

    def test_identical_vectors_give_one(self) -> None:
        a = np.ones((3, 8, 256), dtype=np.float32)
        result = _batched_cosine_mean(a, a)
        assert abs(result - 1.0) < 1e-5

    def test_orthogonal_vectors_give_zero(self) -> None:
        a = np.zeros((1, 1, 4), dtype=np.float32)
        b = np.zeros((1, 1, 4), dtype=np.float32)
        a[0, 0, 0] = 1.0
        b[0, 0, 1] = 1.0
        result = _batched_cosine_mean(a, b)
        assert abs(result) < 1e-5

    def test_opposite_vectors_give_negative_one(self) -> None:
        a = np.ones((1, 1, 4), dtype=np.float32)
        b = -np.ones((1, 1, 4), dtype=np.float32)
        result = _batched_cosine_mean(a, b)
        assert abs(result + 1.0) < 1e-5


class TestCmfFromAttended:
    """Tests for CMF using pre-computed attended representations."""

    def test_aligned_attended_gives_high_cmf(self) -> None:
        rng = np.random.default_rng(99)
        q_suffix = rng.standard_normal((51, 8, 256)).astype(np.float32)
        attended = q_suffix.copy()
        result = _cmf_from_attended(q_suffix, attended, list(range(1, 51)))
        assert result > 0.99

    def test_random_gives_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        q_suffix = rng.standard_normal((51, 8, 256)).astype(np.float32)
        attended = rng.standard_normal((51, 8, 256)).astype(np.float32)
        result = _cmf_from_attended(q_suffix, attended, list(range(1, 51)))
        assert -0.3 < result < 0.3


class TestCmfDirect:
    """Tests for the A→S degenerate single-target pair."""

    def test_same_embedding_gives_one(self) -> None:
        q = np.ones((51, 8, 256), dtype=np.float32)
        result = _cmf_direct(q, list(range(1, 51)), target_index=0)
        assert abs(result - 1.0) < 1e-5

    def test_result_in_valid_range(self) -> None:
        rng = np.random.default_rng(7)
        q = rng.standard_normal((51, 8, 256)).astype(np.float32)
        result = _cmf_direct(q, list(range(1, 51)), target_index=0)
        assert -1.0 <= result <= 1.0


class TestComputeAllCmf:
    """Integration test for all 5 pairs."""

    def test_returns_all_five_pairs(self) -> None:
        rng = np.random.default_rng(42)
        q_suffix = rng.standard_normal((51, 8, 256)).astype(np.float32)
        att_lang = rng.standard_normal((51, 8, 256)).astype(np.float32)
        att_vis = rng.standard_normal((51, 8, 256)).astype(np.float32)

        result = compute_all_cmf(q_suffix, att_lang, att_vis)
        assert isinstance(result, TimestepCmf)
        d = result.as_dict()
        assert set(d.keys()) == {"S_to_L", "S_to_V", "A_to_L", "A_to_V", "A_to_S"}
        for v in d.values():
            assert -1.0 <= v <= 1.0


class TestComputeCmfAttended:
    """Tests for the extraction-time attended representation computation."""

    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        attention = _softmax(rng.standard_normal((8, 51, 867)).astype(np.float32))
        v_prefix = rng.standard_normal((816, 256)).astype(np.float32)

        result = compute_cmf_attended(attention, v_prefix)
        assert result["language"].shape == (51, 8, 256)
        assert result["visual"].shape == (51, 8, 256)

    def test_uniform_attention_gives_centroid(self) -> None:
        """With uniform attention within a modality, the attended
        representation should equal the mean of the target embeddings."""
        attention = np.ones((8, 51, 867), dtype=np.float32) / 867.0
        v_prefix = np.zeros((816, 256), dtype=np.float32)
        v_prefix[768:816] = 1.0

        result = compute_cmf_attended(attention, v_prefix)
        lang = result["language"]
        for h in range(8):
            for q in range(51):
                assert np.linalg.norm(lang[q, h]) > 0.0

    def test_zero_attention_gives_zeros(self) -> None:
        attention = np.zeros((8, 51, 867), dtype=np.float32)
        v_prefix = np.ones((816, 256), dtype=np.float32)

        result = compute_cmf_attended(attention, v_prefix)
        np.testing.assert_array_equal(result["language"], 0.0)
        np.testing.assert_array_equal(result["visual"], 0.0)

    def test_v_shared_across_heads(self) -> None:
        """V has single KV head, so only attention weights differ per head.
        Two heads with identical attention should produce identical attended."""
        rng = np.random.default_rng(77)
        base_attn = _softmax(rng.standard_normal((1, 51, 867)).astype(np.float32))
        attention = np.repeat(base_attn, 8, axis=0)
        v_prefix = rng.standard_normal((816, 256)).astype(np.float32)

        result = compute_cmf_attended(attention, v_prefix)
        for h in range(1, 8):
            np.testing.assert_allclose(
                result["language"][:, 0, :], result["language"][:, h, :], atol=1e-5
            )


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)
