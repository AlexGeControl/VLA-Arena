# Task D2a: V-projection capture + CMF attended representations

> Part of the [Data Analytics epic](../README.md). Extends Track B extraction. Can proceed in parallel with D1.

## Goal

1. Extend `capture.py` to **intercept V-projections** from the attention output einsum.
2. Implement `cmf_attended.py` to compute **intra-modality normalized attended representations** using V-projections.
3. Extend `serialize.py` to write the new **`cmf_attended/`** HDF5 group.
4. Wire everything in `extract_interpret_data.py`.

## Critical pitfall

### PITFALL #15 — V-projection capture counter alignment

The Gemma attention layer executes three operations in sequence:

```
1. BTKGH,BSKH->BKGTS   (logits = Q @ K^T)    counter reads n, captures Q at n, increments to n+1
2. softmax              (attention probs)       counter = n+1, captures probs at n+1
3. BKGTS,BSKH->BTKGH   (output = probs @ V)   counter = n+1, V is args[2]
```

The output einsum fires **after** the counter was incremented by step 1. To align V with Q's key, store V at **`counter - 1`**.

## V-projection capture (`capture.py`)

Add to `CaptureStore`:

```python
self.v_concat: dict[int, jax.Array] = {}
```

Add to `CapturedData`:

```python
v_prefix: dict[int, np.ndarray]   # layer_idx -> [1, 816, 256]
v_suffix: dict[int, np.ndarray]   # layer_idx -> [1, 51, 256]
```

Intercept in `_capturing_einsum`:

```python
elif subscript == "BKGTS,BSKH->BTKGH":
    layer_idx = _store.layer_counter - 1   # align with Q's key
    v_bskh = args[2]                       # [B, S, K=1, H=256]
    v = v_bskh[:, :, 0, :]                 # squeeze KV head -> [B, S, 256]
    def _store_v(arr, idx=layer_idx):
        _store.v_concat[idx] = np.array(arr)
    jax.debug.callback(_store_v, v)
```

In `_extract_results`, split V using the same `q_offset`:

```python
if q_key in _store.v_concat:
    v = _store.v_concat[q_key]
    v_prefix[layer_idx] = v[:, :prefix_len]    # [1, 816, 256]
    v_suffix[layer_idx] = v[:, prefix_len:]     # [1, 51, 256]
```

## CMF attended representations (`cmf_attended.py`)

```python
def compute_cmf_attended(
    attention: np.ndarray,    # [8, 51, 867]
    v_prefix: np.ndarray,     # [816, 256]  — single KV head, batch-squeezed
) -> dict[str, np.ndarray]:
    # Returns {"language": [51, 8, 256], "visual": [51, 8, 256]}
```

For each target modality range `(start, end)`:

1. Extract attention slice: `alpha = attention[:, :, start:end]` → `[8, 51, N]`
2. Normalize within modality: `alpha_norm = alpha / sum(alpha, axis=-1)` → `[8, 51, N]`
3. Get V embeddings: `embed = v_prefix[start:end]` → `[N, 256]`
4. Compute attended: `alpha_norm @ embed` → `[8, 51, 256]` (numpy broadcasts `[8,51,N] @ [N,256]`)
5. Transpose to storage layout: `[51, 8, 256]`

Guard zero-attention modalities with epsilon-safe division.

V-projections are shared across all 8 heads (single KV head). Per-head variation in the attended representation comes **solely** from different attention weight distributions.

## HDF5 serialization (`serialize.py`)

New function `_write_cmf_attended()` writes:

```
/timestep_NNN/cmf_attended/layer_{L:02d}/language   float32 (51, 8, 256)
/timestep_NNN/cmf_attended/layer_{L:02d}/visual     float32 (51, 8, 256)
```

Compression: `gzip`. Written for all 7 sampled layers.

## Extraction orchestrator (`extract_interpret_data.py`)

After capture pass, squeeze V batch dim and pass to `compute_cmf_attended`:

```python
v_prefix_squeezed = {k: v[0] for k, v in captured.v_prefix.items()}  # [816, 256]

for layer_idx in SAMPLED_LAYERS:
    if layer_idx in captured.attention and layer_idx in v_prefix_squeezed:
        cmf_attended_results[layer_idx] = compute_cmf_attended(
            captured.attention[layer_idx],
            v_prefix_squeezed[layer_idx],
        )
```

## Acceptance criteria

- [ ] V capture: `V capture: 18 entries` logged for each timestep (all 18 Gemma layers).
- [ ] V entries use same keys as Q (aligned via `counter - 1`).
- [ ] `CapturedData.v_prefix` shape `[1, 816, 256]`, `v_suffix` shape `[1, 51, 256]` per layer.
- [ ] `compute_cmf_attended` output shapes: `language [51, 8, 256]`, `visual [51, 8, 256]`.
- [ ] With identical attention across all heads, attended representations are identical across heads (V is shared).
- [ ] Zero attention to a modality produces zero-vector attended representation (no NaN).
- [ ] HDF5 contains `cmf_attended/layer_XX/{language,visual}` for all 7 sampled layers.
- [ ] Backend test fixture (`conftest.py`) updated with `cmf_attended` group.
