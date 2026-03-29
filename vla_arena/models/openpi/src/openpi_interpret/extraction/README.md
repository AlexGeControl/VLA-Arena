# Extraction Pipeline (Track B)

Captures attention weights, Q-projections, and pre-computes t-SNE embeddings
and nearest neighbors from the Pi-Zero model. Outputs HDF5 files consumed by
the FastAPI backend (Track C).

## Data Flow

```
VLA-Arena L0 S dataset (HuggingFace)
    ↓  episode frames
Pi-Zero model (JAX, GPU)
    ↓  sample_actions (JIT) → denoised x_0
    ↓  capture pass (disable_jit) → attention probs + Q-projections
scikit-learn t-SNE + scipy KDTree
    ↓  per-layer 2D coords + per-action nearest neighbors
h5py HDF5 writer
    ↓  one .h5 file per episode
backend/data/ (HDF5Reader)
```

## Modules

| File | Purpose |
|------|---------|
| `extract_interpret_data.py` | Main CLI: loads model, iterates episodes/frames, orchestrates pipeline |
| `capture.py` | Monkey-patches `jax.nn.softmax` and `jnp.einsum` to capture attention probs and Q-projections via `jax.debug.callback` |
| `tsne.py` | Head-space Q-projection → t-SNE coordinates + nearest-neighbor pre-computation |
| `serialize.py` | HDF5 writer: attention, t-SNE, neighbors, cameras, metadata |
| `validate.py` | Schema validation for output HDF5 files |

## Key Design Choices

- **Monkey-patching with `jax.debug.callback`**: The Gemma backbone uses `nn.remat` + `nn.scan`, which traces Block internals. Direct Python dict stores would capture JAX tracers instead of concrete values. `jax.debug.callback` materializes arrays from inside traced code.
- **`jax.disable_jit()` for capture pass**: The JIT-compiled `sample_actions` bakes in original `jax.nn.softmax`. Our patches only work in eager mode, so the capture pass runs under `disable_jit()`.
- **Layer counter offset detection**: The backbone forward produces extra einsum calls before the first Gemma layer (from the embedder). The extraction auto-detects the offset from the captured Q-projection keys.

## Usage

```bash
cd <openpi_root>
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  .venv/bin/python src/openpi_interpret/extraction/extract_interpret_data.py \
    --max-episodes 3 --timestep-stride 10
```
