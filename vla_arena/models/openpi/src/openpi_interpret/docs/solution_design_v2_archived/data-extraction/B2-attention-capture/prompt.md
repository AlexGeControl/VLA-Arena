# Task B2: Attention & Hidden State Capture

> Part of the [Data Extraction Pipeline](../README.md) epic. Depends on [B1](../B1-scaffold-model-loading/prompt.md).

## Goal

Capture attention probabilities and Q-projected hidden states from Pi-Zero's backbone during inference, at sampled layers only, using runtime monkey-patching (no edits to `gemma.py`).

## Task

### CaptureStore

Create `openpi_interpret/extraction/capture.py`:

```python
from openpi.models.gemma import Attention, Block

SAMPLED_LAYERS = [0, 3, 6, 9, 12, 15, 17]

class CaptureStore:
    """Storage for captured attention weights and hidden states."""

    def __init__(self):
        self.attention: dict[int, jnp.ndarray] = {}       # layer -> [8, 51, 867]
        self.hidden_states: dict[int, list[jnp.ndarray]] = {}  # layer -> [prefix, suffix]
        self.q_projections: dict[int, list[jnp.ndarray]] = {}  # layer -> [prefix_q, suffix_q]

    def clear(self):
        self.attention.clear()
        self.hidden_states.clear()
        self.q_projections.clear()
```

### Attention Capture via Monkey-Patching

Since Flax NNX modules are mutable Python objects, patch `Attention.__call__` at runtime:

```python
def install_capture_hooks(model, capture_store: CaptureStore):
    """Monkey-patch Attention and Block classes to capture intermediates."""

    original_attn_call = Attention.__call__

    def patched_attention(self, xs, segment_pos, cache, attn_mask):
        result = original_attn_call(self, xs, segment_pos, cache, attn_mask)

        if self._layer_idx in SAMPLED_LAYERS:
            # Extract suffix queries (last 51 rows) attending to all keys (867)
            # probs shape: [B, num_kv_heads, num_q_heads, total_q, total_k]
            # We need: [8, 51, 867] for each sampled layer
            suffix_probs = probs[:, :, :, prefix_len:, :]
            capture_store.attention[self._layer_idx] = suffix_probs

            # Capture Q projections for t-SNE (Task B3)
            # q shape: [B, total, num_heads, head_dim]
            capture_store.q_projections[self._layer_idx] = [q_prefix, q_suffix]

        return result

    Attention.__call__ = patched_attention
```

### Implementation Notes

The exact patching strategy depends on `Attention.__call__`'s local variables. The implementer must read the actual `gemma.py` source (after submodule init) to determine whether `probs` is accessible. The pseudocode above shows the intent; the actual implementation may need to:

1. Store Q and K as module attributes before softmax, or
2. Duplicate the `jnp.einsum + softmax` computation from the already-projected Q and K, or
3. Use a JAX `custom_vjp` or `io_callback` to intercept the value

The key constraint is: **only modify runtime behavior, never edit `gemma.py` source files**.

### Hidden State Capture

For t-SNE, also capture post-attention hidden states per block:

```python
original_block_call = Block.__call__

def patched_block(self, xs, segment_pos, cache, attn_mask):
    result = original_block_call(self, xs, segment_pos, cache, attn_mask)

    if self._layer_idx in SAMPLED_LAYERS:
        capture_store.hidden_states[self._layer_idx] = [
            result[0],  # prefix hidden state [B, 816, 2048]
            result[1],  # suffix hidden state [B, 51, 1024]
        ]

    return result

Block.__call__ = patched_block
```

### Output File

```
openpi_interpret/extraction/capture.py
```

## Acceptance Criteria

- [ ] After one `sample_actions` call, `capture_store.attention` has 7 entries (one per sampled layer)
- [ ] Each entry has shape `[8, 51, 867]` (heads × suffix queries × total keys)
- [ ] Attention rows sum to ~1.0 (valid softmax output)
- [ ] `capture_store.q_projections` has 7 entries with correct shapes
- [ ] Model inference output is unchanged (patches are side-effect-free)
