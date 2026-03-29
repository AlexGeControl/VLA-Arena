# Task B5: End-to-end extraction + validation + backend smoke

> Part of the [Data Extraction epic](../README.md). Depends on [B4](../B4-hdf5-serialization/prompt.md).

## Goal

Run the **full pipeline** on a small real slice of data, **validate** HDF5 outputs, and confirm the **FastAPI backend** can serve every endpoint against the new files.

## Critical pitfalls

### PITFALL #6 — GPU OOM across many forwards

Insert **after every timestep** and **after every episode** (and optionally before heavy CPU phases):

```python
import gc
import jax

gc.collect()
jax.clear_caches()
```

If the codebase exposes extra JAX / backend cache flushes, call those too. The objective is **stable peak memory** across **3 episodes × ~11–13 timesteps** at stride 10.

## Run configuration

- **Episodes**: **`--max-episodes 3`**
- **Stride**: **`--timestep-stride 10`**
- **Checkpoint / dataset**: same defaults as B1 (local snapshot + `local_files_only` where applicable).
- **Environment**: OpenPI **`uv`** venv, `CUDA_VISIBLE_DEVICES=0`, `XLA_PYTHON_CLIENT_ALLOCATOR=platform` recommended.

## Validation (`validate.py`)

Implement checks (automated **`pytest`** + CLI entrypoint optional):

| Check | Rule |
|-------|------|
| Attention shape | For each `timestep_*`, each `attention/layer_*`: **`(8, 51, 867)`** |
| Attention rows | For each head-query row, sum ≈ **1.0** (`atol=1e-3` or tighter if stable) |
| t-SNE | All values **finite**; shape **`(867, 2)`** |
| Neighbors | Shape **`(50, 5)`**; `neighbor_index` in **`[0, 866]`**; distances **non-negative** |
| Token meta | JSON length **867**; every image patch token has **`type == "image_patch"`** |
| Cameras | Under **`timestep_*/cameras/`**, datasets **`base_0_rgb`**, **`left_wrist_0_rgb`**, **`right_wrist_0_rgb`** exist with **`uint8`** shape **`(224, 224, 3)`** |
| Meta attrs | `episode_id`, `task_instruction`, `num_timesteps`, `instruction_tokens`, `sampled_layers` present; `sampled_layers` parses to **`[0,3,6,9,12,15,17]`** |

```python
# Example: attention row sums
attn = f["timestep_000"]["attention"]["layer_00"][()]
sums = attn.sum(axis=-1)  # (8, 51)
assert np.allclose(sums, 1.0, atol=1e-3)
```

## Backend smoke test

1. Point **`EpisodeIndex`** / backend config at the directory containing the three **`.h5`** files.
2. Start **uvicorn** (or project script): e.g. `uvicorn app.main:app --host 0.0.0.0 --port 8000` from the backend package root.
3. **GET** every public route the frontend uses, including (adjust paths to match `main.py`):

   - Episode list / metadata
   - Timestep list / resolve timestep key
   - Attention row or matrix slice for a head/layer/action
   - t-SNE points for a layer
   - Neighbors for a layer + action index
   - Camera PNG or raw image endpoint per timestep

4. Expect **200** responses; JSON fields match Pydantic models (no missing keys for the UI).

## Acceptance criteria

- [ ] Full extraction completes for **3 episodes**, stride **10**, without OOM (PITFALL #6 mitigations in place).
- [ ] **`validate.py`** passes on **all** produced `.h5` files.
- [ ] Backend starts against the output directory; **all** documented endpoints return **200** with sensible shapes.
- [ ] README or `--help` documents the exact command line for reproduction.
- [ ] At least one **`pytest`** that runs **`validate`** on a **tiny fixture** `.h5` in CI (optional GPU-less path).
