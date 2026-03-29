---
name: extract-model-states
description: >-
  Run the Pi-Zero attention extraction pipeline on VLA-Arena episodes.
  Produces HDF5 files with attention weights, t-SNE coordinates, and nearest
  neighbors. Use when asked to extract data, re-extract with different
  parameters, or populate the data/ directory.
---

# Extract Model States

## Prerequisites

- OpenPI `uv` venv is synced: `cd <openpi_root> && uv sync`
- scikit-learn installed: `uv pip install scikit-learn`
- GPU available (7 GB VRAM minimum)
- Pi-Zero checkpoint cached at `~/.cache/huggingface/hub/models--VLA-Arena--pi0-vla-arena-fintuned/snapshots/acdc8e7eaa6dfccedef6db26626ec828bfa21b1e`

## Run Extraction

```bash
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  .venv/bin/python src/openpi_interpret/extraction/extract_interpret_data.py \
    --max-episodes 3 \
    --timestep-stride 10 \
    --output-dir src/openpi_interpret/data
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | Known-good HF cache snapshot | Local checkpoint directory |
| `--dataset-repo-id` | `VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla` | HF dataset (must be cached) |
| `--max-episodes` | 3 | Number of episodes to extract |
| `--timestep-stride` | 10 | Sample every Nth frame (10 = 1.0s at 10Hz) |
| `--output-dir` | `src/openpi_interpret/data` | Where to write HDF5 files |

## Expected Output

- One `.h5` file per episode (~300-400 MB each with stride 10)
- ~11-13 timesteps per episode
- Each timestep contains attention (7 layers x 8 heads), t-SNE coords, nearest neighbors, camera images

## Timing

- ~12s per timestep (inference + capture + t-SNE)
- ~3 min per episode
- ~10 min total for 3 episodes

## Pitfalls

- Do NOT use `snapshot_download` — it may pick a corrupted snapshot
- The script uses `local_files_only=True` — all HF artifacts must be pre-cached
- GPU OOM after many timesteps: the script calls `gc.collect()` + `jax.clear_caches()` between timesteps automatically
