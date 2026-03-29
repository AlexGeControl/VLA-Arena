"""Monkey-patch hooks that capture attention probabilities and Q-projections from Pi-Zero.

Run extraction with JAX-friendly env (same as ``extract_interpret_data.py``)::

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform HF_HUB_OFFLINE=1

Strategy:
  1. Before a manual forward pass, replace ``jax.nn.softmax`` with a wrapper
     that stores its output keyed by an auto-incrementing layer counter.
  2. Similarly intercept the concatenated Q tensor right after RoPE + scaling.
  3. After the pass, restore the originals and return captured data.

Because the manual forward pass runs **outside** ``jax.lax.while_loop``, Python
side-effects (dict writes) work normally.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import einops
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0

logger = logging.getLogger(__name__)

SAMPLED_LAYERS: list[int] = [0, 3, 6, 9, 12, 15, 17]


@dataclasses.dataclass
class CapturedData:
    """Attention and Q-projection tensors extracted from a single forward pass."""

    attention: dict[int, np.ndarray]
    q_prefix: dict[int, np.ndarray]
    q_suffix: dict[int, np.ndarray]
    prefix_len: int
    suffix_len: int


class CaptureStore:
    """Mutable container reset before each forward pass."""

    def __init__(self) -> None:
        self.layer_counter: int = 0
        self.probs: dict[int, jax.Array] = {}
        self.q_pre_gqa: dict[int, jax.Array] = {}

    def reset(self) -> None:
        self.layer_counter = 0
        self.probs.clear()
        self.q_pre_gqa.clear()


_store = CaptureStore()
_original_softmax = jax.nn.softmax
_original_einsum = jnp.einsum
_patched = False


def _capturing_softmax(x: jax.Array, *, axis: int = -1) -> jax.Array:
    """Drop-in replacement for ``jax.nn.softmax`` that records attention probs.

    Only captures 5D tensors ``[B, K, G, T, S]`` (Gemma backbone attention).
    Skips 4D tensors from the SigLIP vision encoder.
    Stores ALL layers, not just sampled — the layer counter tracks which
    Gemma backbone layer we're in.
    """
    result = _original_softmax(x, axis=axis)
    if x.ndim == 5:
        layer_idx = _store.layer_counter
        def _store_probs(arr: jax.Array, idx: int = layer_idx) -> None:
            _store.probs[idx] = np.array(arr)
        jax.debug.callback(_store_probs, result)
    return result


def _capturing_einsum(*args, **kwargs):
    """Drop-in for ``jnp.einsum`` that intercepts the Q·K logits call to grab Q.

    Captures ALL layers and increments the counter on every ``BTKGH,BSKH->BKGTS`` call.
    Sampled-layer filtering happens in ``_extract_results``.
    """
    result = _original_einsum(*args, **kwargs)
    if len(args) >= 3 and isinstance(args[0], str):
        subscript = args[0]
        if subscript == "BTKGH,BSKH->BKGTS":
            layer_idx = _store.layer_counter
            q_btkgh = args[1]
            q = einops.rearrange(q_btkgh, "B T K G H -> B T (K G) H")
            def _store_q(arr: jax.Array, idx: int = layer_idx) -> None:
                _store.q_pre_gqa[idx] = np.array(arr)
            jax.debug.callback(_store_q, q)
            _store.layer_counter += 1
    return result


def install_hooks() -> None:
    """Replace ``jax.nn.softmax`` and ``jnp.einsum`` with capturing versions."""
    global _patched  # noqa: PLW0603
    if _patched:
        return
    jax.nn.softmax = _capturing_softmax  # type: ignore[assignment]
    jnp.einsum = _capturing_einsum  # type: ignore[assignment]
    _patched = True
    logger.info("Capture hooks installed")


def uninstall_hooks() -> None:
    """Restore original functions."""
    global _patched  # noqa: PLW0603
    jax.nn.softmax = _original_softmax  # type: ignore[assignment]
    jnp.einsum = _original_einsum  # type: ignore[assignment]
    _patched = False
    logger.info("Capture hooks removed")


def run_capture_pass(
    model: "Pi0",
    observation,
    actions_x0: jax.Array,
    timestep: float = 0.1,
) -> CapturedData:
    """Run a single non-JIT forward pass through the backbone with capture hooks active.

    Args:
        model: Loaded Pi0 model instance.
        observation: Preprocessed ``Observation`` (batch size 1).
        actions_x0: Denoised action tensor ``[1, action_horizon, action_dim]``.
        timestep: Denoising timestep for suffix embedding (small value near 0).

    Returns:
        CapturedData with attention maps and Q-projections for sampled layers.
    """
    from openpi.models.pi0 import make_attn_mask

    install_hooks()
    _store.reset()

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    t = jnp.broadcast_to(jnp.array(timestep), (actions_x0.shape[0],))
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, actions_x0, t
    )

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    # Reset counter just before the backbone forward — embed_prefix may have
    # triggered einsum calls (e.g. from the SigLIP encoder or embedder) that
    # incremented the layer counter prematurely.
    _store.reset()

    with jax.disable_jit():
        model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

    prefix_len = int(prefix_tokens.shape[1])
    suffix_len = int(suffix_tokens.shape[1])

    return _extract_results(prefix_len, suffix_len)


def _extract_results(prefix_len: int, suffix_len: int) -> CapturedData:
    """Slice captured tensors to suffix-query attention and split Q-projections.

    The backbone forward produces extra einsum/softmax calls before the first
    Gemma layer (from the embedder or initial projection). We detect the
    offset automatically: the Q dict should have exactly 18 entries (one per
    Gemma layer). The smallest key in Q corresponds to Gemma layer 0.
    """
    attention: dict[int, np.ndarray] = {}
    q_prefix: dict[int, np.ndarray] = {}
    q_suffix: dict[int, np.ndarray] = {}

    if not _store.q_pre_gqa:
        logger.warning("No Q-projections captured — returning empty CapturedData")
        return CapturedData(attention={}, q_prefix={}, q_suffix={},
                            prefix_len=prefix_len, suffix_len=suffix_len)

    q_keys = sorted(_store.q_pre_gqa.keys())
    q_offset = q_keys[0]
    logger.info("Q capture offset=%d (keys %d–%d for 18 layers)", q_offset, q_keys[0], q_keys[-1])

    probs_keys = sorted(_store.probs.keys())
    probs_offset = probs_keys[0] if probs_keys else q_offset + 1
    logger.info("Probs capture offset=%d (keys %d–%d)", probs_offset, probs_keys[0] if probs_keys else -1, probs_keys[-1] if probs_keys else -1)

    for layer_idx in SAMPLED_LAYERS:
        probs_key = layer_idx + probs_offset
        if probs_key in _store.probs:
            probs = _store.probs[probs_key]
            suffix_attn = _slice_suffix_attention(probs, prefix_len, suffix_len)
            attention[layer_idx] = suffix_attn

        q_key = layer_idx + q_offset
        if q_key in _store.q_pre_gqa:
            q = _store.q_pre_gqa[q_key]
            q_prefix[layer_idx] = q[:, :prefix_len]
            q_suffix[layer_idx] = q[:, prefix_len:]

    logger.info("Extracted %d attention layers, %d Q-projection layers", len(attention), len(q_prefix))

    return CapturedData(
        attention=attention,
        q_prefix=q_prefix,
        q_suffix=q_suffix,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
    )


def _slice_suffix_attention(
    probs: np.ndarray, prefix_len: int, suffix_len: int
) -> np.ndarray:
    """Extract suffix-query rows from full attention probs.

    Args:
        probs: Shape ``[B, K, G, total_q, total_k]`` (K=1 for GQA).
        prefix_len: Number of prefix tokens.
        suffix_len: Number of suffix tokens.

    Returns:
        Array of shape ``[num_heads, suffix_len, total_seq]`` (batch dim squeezed).
    """
    total_seq = prefix_len + suffix_len
    suffix_probs = probs[0, :, :, prefix_len:total_seq, :total_seq]
    return einops.rearrange(suffix_probs, "K G S T -> (K G) S T")
