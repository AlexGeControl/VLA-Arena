# Task B2: Attention probabilities + Q-projections (monkey-patch capture)

> Part of the [Data Extraction epic](../README.md). Depends on [B1](../B1-scaffold-model-loading/prompt.md).

## Goal

Capture **attention probability tensors** and **Q-projections** (prefix and suffix) for **sampled Gemma layers** by **monkey-patching** JAX primitives during a dedicated capture forward pass.

## Critical pitfalls

### PITFALL #2 — `jax.lax.while_loop` / `sample_actions`

- **Do not** use `sample_actions` (or any path that stays inside a traced `while_loop`) as the container for patch side effects you need as concrete arrays.
- **Procedure**:
  1. Run **`sample_actions` first with normal JIT** to obtain the final denoised state / `x_0` (or equivalent) for the timestep.
  2. Run a **second**, **separate forward** that feeds that state into the backbone / attention path with **`jax.disable_jit()`** while patches are active.

### PITFALL #3 — `nn.remat` tracing

- Tensors living only inside rematerialized blocks may not be assignable to a Python `dict` from trace time.
- Inside `_capturing_softmax` and `_capturing_einsum`, push host-visible data with **`jax.debug.callback(callback_fn, *arrays)`** so values are **materialized** outside the remat trace.

Example pattern:

```python
def _store_attention(host_array: np.ndarray) -> None:
    store.append(np.asarray(host_array))

def capturing_softmax(x, axis=-1):
    y = orig_softmax(x, axis=axis)
    if should_capture(y):
        jax.debug.callback(_store_attention, y)
    return y
```

### PITFALL #4 — SigLIP vs Gemma

- **Only** capture softmax outputs with **`ndim == 5`**: shape **`[B, K, G, T, S]`** (Gemma backbone attention).
- **Skip** **`ndim == 4`** softmax outputs (SigLIP ViT); they will poison layer indexing and memory.

### PITFALL #5 — Layer counter offset

- The backbone executes **extra** `einsum` / `softmax` work **before** the first Gemma self-attention layer you care about.
- **Reset** your internal `_store` / layer counter **immediately before** entering the Gemma backbone forward you instrument.
- **Auto-detect** the mapping from capture index → logical Gemma layer by inspecting **Q-related keys** (e.g. minimum stored Q key corresponds to **Gemma layer 0** after reset).

## Implementation notes

### Global patches

1. Save originals: `orig_softmax = jax.nn.softmax`, `orig_einsum = jnp.einsum`.
2. Replace **`jax.nn.softmax`** and **`jnp.einsum`** with wrappers that:
   - Call the original.
   - Optionally record outputs / intermediates per rules below.

### Identifying Q·K logits (`einsum`)

The **query–key matmul** in this stack uses an einsum of the form:

```text
"BTKGH,BSKH->BKGTS"
```

(Verify exact string in model code; subscripts may use equivalent ordering.)

- On each match, **increment** a **per-forward** layer counter (after your pre-backbone reset).
- Use this counter to bucket subsequent softmax tensors into **logical layers**.

### Softmax → attention probs

After filtering **5D** tensors, reshape / reduce to the stored contract **`(8, 51, 867)`**: **8 heads**, **51** suffix queries, **867** keys (prefix + suffix).

Ensure **rows** (per head, per query) **sum to 1** (within float tolerance).

## Data container

Define a **`CapturedData`** dataclass (name may match `capture.py`):

```python
@dataclass
class CapturedData:
    attention: dict[int, np.ndarray]      # layer_idx -> (8, 51, 867) float32
    q_prefix: dict[int, np.ndarray]         # layer_idx -> (816, 8, 256)
    q_suffix: dict[int, np.ndarray]         # layer_idx -> (51, 8, 256)
    prefix_len: int                         # 816
    suffix_len: int                         # 51
```

Populate **`attention`** only for **sampled** layers `[0, 3, 6, 9, 12, 15, 17]` after mapping physical indices with the auto-detected offset.

## File placement

Implement in **`openpi_interpret/extraction/capture.py`**:

- `install_capture_hooks()` / `remove_capture_hooks()` (context manager recommended).
- `run_capture_forward(policy, obs, x_0_or_state, ...)` — uses **`jax.disable_jit()`** and returns `CapturedData`.

## Acceptance criteria

- [ ] No capture logic inside traced `sample_actions` / `while_loop` body; **two-phase** JIT inference + JIT-off capture.
- [ ] `jax.debug.callback` used for stores from softmax / einsum paths that run under **remat**.
- [ ] Only **5D** softmax tensors recorded; **4D** SigLIP paths skipped.
- [ ] Layer counter reset + **offset auto-detection** documented in code comments (one short paragraph).
- [ ] For each sampled layer present: **`attention[layer].shape == (8, 51, 867)`**, `float32`, each row sums to **~1.0** (e.g. `np.allclose(row_sum, 1.0, atol=1e-3)`).
- [ ] Q tensors: **`q_prefix` shape `(816, 8, 256)`**, **`q_suffix` shape `(51, 8, 256)`** per sampled layer.
