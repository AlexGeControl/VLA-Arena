# Track B — Data Extraction Epic (OpenPI InterpreT)

Part of [Solution Design v3](../README.md). This epic produces **Contract 1** artifacts: HDF5 files the FastAPI backend reads to serve attention, t-SNE, neighbors, and cameras to the frontend.

## Purpose

Extract, from a fine-tuned **Pi-Zero** policy at inference time:

- **Attention probabilities** (per sampled Gemma layer, per head): suffix queries × full prefix+suffix keys.
- **Q-projections** in head space (prefix and suffix token positions).
- **t-SNE** coordinates in that shared 2048-D head space (867 tokens × 2 per layer).
- **Nearest neighbors** from each of 50 action tokens into five modality groups.

All heavy computation runs **offline** during extraction; the backend stays ML-free.

## Architecture (end-to-end)

```
Pi-Zero inference (JIT)
       → obtain denoised state / x_0 for the timestep
Separate forward / capture pass (jax.disable_jit())
       → monkey-patched jax.nn.softmax + jnp.einsum
       → jax.debug.callback to escape nn.remat tracing
t-SNE + KDTree neighbors (CPU, sklearn / scipy)
       → per-timestep bundles
HDF5 serialization (h5py)
       → one file per episode
```

1. **Pi-Zero inference**: Normal policy forward to run the model as shipped (including `sample_actions` / flow sampling).
2. **Monkey-patched capture**: Global patches on `jax.nn.softmax` and `jnp.einsum` record Gemma attention and Q-related tensors without re-entering traced `while_loop` bodies incorrectly.
3. **t-SNE pre-computation**: Fit `sklearn.manifold.TSNE` on concatenated prefix+suffix Q-features per sampled layer.
4. **HDF5 serialization**: Hierarchical layout with **per-timestep** groups and **per-timestep** camera images.

## Task table

| Task | Focus | Key deliverables | Pitfalls to respect |
|------|--------|------------------|---------------------|
| **B1** | Scaffold + model I/O | CLI, `create_trained_policy`, one successful inference, dataset parquet/metadata via HF | #1 cache path, #12 `policy._model` |
| **B2** | Attention + Q capture | Patches, `CapturedData`, layer indexing, 5D-only softmax | #2 `while_loop`, #3 `remat`, #4 SigLIP 4D, #5 layer offset |
| **B3** | t-SNE + neighbors | `(867, 2048)` features, TSNE, KDTree per modality | Index ranges for 5 groups |
| **B4** | HDF5 writer | `write_episode_hdf5`, full schema, dtypes | #7 `image_patch`, #9 cameras under each timestep |
| **B5** | E2E + QA | 3 episodes, stride 10, `validate.py`, backend smoke | #6 GPU memory |

## Known pitfalls (battle-tested)

| # | Issue | Mitigation |
|---|--------|------------|
| **1** | Hugging Face checkpoint / snapshot corruption | Use the **direct cache path** to the known-good snapshot, e.g. `~/.cache/huggingface/hub/models--VLA-Arena--pi0-vla-arena-fintuned/snapshots/acdc8e7eaa6dfccedef6db26626ec828bfa21b1e`, with `local_files_only=True`. **Do not** use `snapshot_download` for this checkpoint. |
| **2** | `jax.lax.while_loop` tracing | **Do not** install capture patches inside `sample_actions` / the sampling loop in a way that runs under that trace. Run **`sample_actions` first (JIT)** to get `x_0`, then run a **separate capture pass** with **`jax.disable_jit()`**. |
| **3** | `nn.remat` / gradient checkpointing | Values computed inside remat-traced code may not be storable in plain Python dicts. Use **`jax.debug.callback`** inside patched `_capturing_softmax` / `_capturing_einsum` to **materialize** host-side NumPy arrays. |
| **4** | SigLIP softmax interference | Only record softmax outputs that are **5D**: `[B, K, G, T, S]` (Gemma backbone). **Skip 4D** tensors (SigLIP ViT path). |
| **5** | Layer counter offset | The backbone forward executes **extra** `einsum` / `softmax` ops **before** the first real Gemma layer. **Reset** the capture store counter **immediately before** backbone forward, then **auto-detect** the Gemma layer index offset from captured Q-key ranges (e.g. **minimum Q key = physical layer 0**). |
| **6** | GPU OOM | After **each timestep** and **each episode**, call **`gc.collect()`** and **`jax.clear_caches()`** (and any project-specific cache clears) so XLA does not retain unbounded device memory. |
| **7** | Token type naming | In `build_token_meta()`, vision patch entries must use **`"image_patch"`**, not **`"image"`**. The backend and frontend assume this contract. |

## Environment

- **Python**: OpenPI project **`uv`** virtualenv (see baseline `README.md`).
- **GPU**: e.g. `CUDA_VISIBLE_DEVICES=0`.
- **Allocator** (recommended for long JAX runs): `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.

## Sampled layers

Use this list everywhere (extraction, HDF5 meta, backend):

`[0, 3, 6, 9, 12, 15, 17]`

## HDF5 output schema (Contract 1 summary)

One file: `{episode_id}.h5`.

- **`/meta`** (attributes): `episode_id`, `task_instruction`, `num_timesteps`, `instruction_tokens` (JSON string), `sampled_layers` (JSON string).
- **`/timestep_{frame_index}/`** — `frame_index` is the **dataset frame index** (zero-padded width as implemented, e.g. `timestep_000`, `timestep_010`; not necessarily 0..N-1 sequential-only).
  - **`cameras/`** — `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`: `uint8`, shape `(224, 224, 3)`. **Per timestep**, not at episode root.
  - **`token_meta`** — JSON string, **867** objects; image patches use **`type: "image_patch"`**.
  - **`attention/layer_XX`** — `float32`, `(8, 51, 867)`.
  - **`tsne/layer_XX`** — `float32`, `(867, 2)`.
  - **`neighbors/layer_XX`** — compound `(50, 5)` with fields `neighbor_index` (`<i4`), `distance` (`<f4`).
  - **`q_projections/layer_XX/prefix`** — `(816, 8, 256)`; **`suffix`** — `(51, 8, 256)`.

Full field-level detail: [B4-hdf5-serialization/prompt.md](B4-hdf5-serialization/prompt.md).

## Code layout (extraction package)

Under `openpi_interpret/extraction/`:

| Module | Role |
|--------|------|
| `extract_interpret_data.py` | CLI orchestration: episodes, stride, inference + capture + serialize |
| `capture.py` | Patches, `CapturedData`, capture pass helpers |
| `tsne.py` | Head-space features, TSNE, KDTree neighbors |
| `serialize.py` | `write_episode_hdf5` and helpers |
| `validate.py` | Schema and numeric sanity checks on written `.h5` files |

## Task prompts

1. [B1 — Scaffold & model loading](B1-scaffold-model-loading/prompt.md)
2. [B2 — Attention capture](B2-attention-capture/prompt.md)
3. [B3 — t-SNE & neighbors](B3-tsne-neighbors/prompt.md)
4. [B4 — HDF5 serialization](B4-hdf5-serialization/prompt.md)
5. [B5 — Validation & smoke tests](B5-validation/prompt.md)
