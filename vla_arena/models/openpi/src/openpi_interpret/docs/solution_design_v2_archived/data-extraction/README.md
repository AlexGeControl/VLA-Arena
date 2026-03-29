# Epic: Data Extraction Pipeline

> **Track B** of the OpenPI InterpreT parallel implementation plan.

This epic produces pre-processed HDF5 data files that the FastAPI backend (Track C) serves to the frontend. It runs Pi-Zero inference on episodes from the VLA-Arena L0 S dataset, captures attention weights and hidden states via runtime monkey-patching, pre-computes t-SNE projections and nearest neighbors, and serializes everything to HDF5.

**This epic's output is consumed by the backend (Track C), not directly by the frontend.**

## Architecture Facts

From [model_architecture.md](../../../../../../docs/pi-zero/model_architecture.md):

- Pi-Zero's backbone has **18 layers**, each with **8 attention heads** (1 KV head, GQA with 8 groups).
- The unified token sequence has **~867 tokens**: 768 image patches (3 cameras × 256) + up to 48 language tokens + 1 state token + 50 action tokens.
- Two experts share the backbone: **Expert 0** (PaliGemma, width 2048) processes the prefix (image + language), **Expert 1** (Action, width 1024) processes the suffix (state + actions).
- Cross-expert attention is computed in **shared head space** (dim 256 per head, 8 heads).
- Attention probabilities are computed in `gemma.py` `Attention.__call__` via `jax.nn.softmax` — an intermediate not returned by default.

## Design Decision: Standalone In-Process Script

We create a standalone Python script that:

1. Loads the Pi-Zero model **in-process** (same config + checkpoint as `serve_policy.py`)
2. Reuses the existing data transform chain
3. Runs inference with **runtime monkey-patched attention capture** (no edits to `gemma.py`)
4. **Pre-computes t-SNE and nearest neighbors** (no ML deps needed at backend runtime)
5. Serializes to **HDF5** via `h5py`

## Dataset

| Parameter | Value |
|-----------|-------|
| Dataset | `VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla` |
| Episodes | 602 total, select 10 for first attempt (configurable) |
| Frames/episode | ~120 average |
| Timestep stride | Every ~30 frames → ~4 timesteps per episode |
| Tasks | 60 unique language instructions |

## Sampled Layers

7 of 18 layers (stride 3): `[0, 3, 6, 9, 12, 15, 17]`.

## Output Format: HDF5

One file per episode. See [B4 Serialization](B4-hdf5-serialization/prompt.md) for the complete schema.

```
{episode_id}.h5
  /meta                    (attrs: episode_id, task_instruction, ...)
  /cameras/{name}          (uint8 [H, W, 3])
  /timestep_{t:03d}/
    /token_meta            (JSON string, 867 entries)
    /attention/layer_{ll}  (float32 [8, 51, 867], chunked)
    /tsne/layer_{ll}       (float32 [867, 2], pre-computed)
    /neighbors/layer_{ll}  (compound [50, 5], pre-computed)
    /q_projections/layer_{ll}/{prefix,suffix}  (float32, backlog asset)
```

## Data Budget

| Parameter | Value |
|-----------|-------|
| Per-timestep attention (float32) | 7 × 8 × 51 × 867 × 4 = **~10 MB** |
| Per-timestep t-SNE | 7 × 867 × 2 × 4 = **~0.05 MB** |
| Per-timestep neighbors | 7 × 50 × 5 × 8 = **~0.01 MB** |
| Per-timestep Q-projections | 7 × (816+51) × 8 × 256 × 4 = **~50 MB** |
| **Total for 10 episodes (~40 timesteps)** | **~2.4 GB** (dominated by Q-projections) |

> **Note**: Q-projections are a backlog asset. If storage is a concern, they can be omitted, reducing the budget to ~400 MB for 10 episodes.

## Tasks

| Task | Deliverable | Dependency |
|------|------------|------------|
| [B1. Scaffold & Model Loading](B1-scaffold-model-loading/prompt.md) | CLI script, model loading, single-sample inference verification | VLA-Arena submodule + `uv sync` |
| [B2. Attention & Hidden State Capture](B2-attention-capture/prompt.md) | Monkey-patched capture hooks for attention probs + Q-projections | B1 |
| [B3. t-SNE & Neighbors](B3-tsne-neighbors/prompt.md) | Head-space t-SNE + nearest-neighbor pre-computation per action/modality | B2 |
| [B4. Serialization](B4-hdf5-serialization/prompt.md) | HDF5 writer (`h5py`), float32 attention, pre-computed t-SNE + neighbors | B2 + B3 |
| [B5. Validation](B5-validation/prompt.md) | End-to-end run on 2–3 episodes, HDF5 format verification, backend smoke test | B4 |

```
B1 ──► B2 ──► B3 ──┐
                    ├──► B4 ──► B5
              B2 ───┘
```

## Environment & Prerequisites

**Runtime**: OpenPI `uv` environment:

```bash
cd vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi
uv sync
uv pip install scikit-learn h5py
```

**Checkpoint**: Same as [metrics/README.md](../../../../../../docs/pi-zero/metrics/README.md).

**GPU**: ~7 GB VRAM.

## File Placement

```
openpi_interpret/
  extraction/
    extract_interpret_data.py     # Main script (B1)
    capture.py                    # Capture hooks (B2)
    tsne.py                       # t-SNE + neighbors (B3)
    serialize.py                  # HDF5 writer (B4)
    validate.py                   # Validation (B5)
  data/                           # HDF5 output (gitignored)
```
