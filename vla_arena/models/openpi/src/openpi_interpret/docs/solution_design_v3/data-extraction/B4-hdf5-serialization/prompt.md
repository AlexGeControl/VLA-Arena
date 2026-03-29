# Task B4: HDF5 serialization (Contract 1)

> Part of the [Data Extraction epic](../README.md). Depends on [B2](../B2-attention-capture/prompt.md) and [B3](../B3-tsne-neighbors/prompt.md).

## Goal

Write **`write_episode_hdf5()`** in **`openpi_interpret/extraction/serialize.py`** so that each episode produces one **`.h5`** file the backend’s **`HDF5Reader`** can open **read-only** without layout surprises.

## Critical pitfalls

### PITFALL #7 — Token type string

- In **`build_token_meta()`** (or equivalent), vision patch entries **must** use **`"type": "image_patch"`**.
- **Never** use `"image"` alone for patch rows; downstream parsers treat that as a contract violation.

### PITFALL #9 — Camera storage location

- Camera RGB tensors **must** live under **each timestep group**:

  **`/timestep_{frame_index}/cameras/{camera_name}`**

- **Do not** store a single episode-level `/cameras/` only (robot moves; per-timestep images are required).

## Full HDF5 schema

File: **`{output_dir}/{episode_id}.h5`**

### `/meta` (HDF5 group attributes)

| Attribute | Type | Description |
|-----------|------|-------------|
| `episode_id` | `str` | Filename stem / episode key |
| `task_instruction` | `str` | Natural language task |
| `num_timesteps` | `int` | Number of `timestep_*` groups |
| `instruction_tokens` | `str` | JSON list of token strings |
| `sampled_layers` | `str` | JSON list of ints: `[0,3,6,9,12,15,17]` |

### `/timestep_{frame_index}/` (one group per retained frame)

`frame_index` = **dataset frame index** (e.g. `0`, `10`, `20` → groups `timestep_000`, `timestep_010`, `timestep_020` if using zero-padded width 3; **be consistent** with backend `_resolve_ts_key` expectations).

#### `cameras/` (group)

| Dataset | Dtype | Shape |
|---------|-------|-------|
| `base_0_rgb` | `uint8` | `(224, 224, 3)` |
| `left_wrist_0_rgb` | `uint8` | `(224, 224, 3)` |
| `right_wrist_0_rgb` | `uint8` | `(224, 224, 3)` — zeros if masked / missing |

#### Root of timestep group

| Dataset | Type | Content |
|---------|------|---------|
| `token_meta` | variable-length or UTF-8 string | JSON array length **867**; each element includes at least `index`, `type` (`image_patch` for patches), modality metadata as defined by backend |

#### `attention/` (group)

For each sampled layer `L` in `{0,3,6,9,12,15,17}`:

| Dataset | Dtype | Shape | Notes |
|---------|-------|-------|--------|
| `layer_{L:02d}` | `float32` | `(8, 51, 867)` | Optional `chunks` e.g. `(1, 51, 867)`; `compression="gzip"` acceptable |

#### `tsne/` (group)

| Dataset | Dtype | Shape |
|---------|-------|-------|
| `layer_{L:02d}` | `float32` | `(867, 2)` |

#### `neighbors/` (group)

Compound dtype:

```python
NEIGHBOR_DTYPE = np.dtype([("neighbor_index", "<i4"), ("distance", "<f4")])
```

| Dataset | Dtype | Shape |
|---------|-------|-------|
| `layer_{L:02d}` | `NEIGHBOR_DTYPE` | `(50, 5)` |

#### `q_projections/` (group)

| Path | Dtype | Shape |
|------|-------|-------|
| `layer_{L:02d}/prefix` | `float32` | `(816, 8, 256)` |
| `layer_{L:02d}/suffix` | `float32` | `(51, 8, 256)` |

## Function signature

```python
from pathlib import Path

def write_episode_hdf5(
    output_dir: Path,
    episode_id: str,
    task_instruction: str,
    instruction_tokens: list[str],
    timestep_data: list[dict],
) -> Path:
    """Write one episode HDF5 file.

    Each element of *timestep_data* must include:
      - ``timestep``: int frame index (used to build ``timestep_{idx:03d}`` or agreed padding)
      - ``token_meta``: list[dict] (867 entries, image_patch types)
      - ``attention``: dict[int, ndarray]  # (8, 51, 867) float32
      - ``tsne``: dict[int, ndarray]       # (867, 2) float32
      - ``neighbors``: dict[int, ndarray] # (50, 5) NEIGHBOR_DTYPE
      - ``camera_images``: dict[str, ndarray]  # uint8 (224,224,3), written under timestep/cameras/
      - ``q_prefix`` / ``q_suffix``: dict[int, ndarray] (optional but recommended for backlog / debug)

    Returns:
        Path to ``{episode_id}.h5``.
    """
```

Adapt key names to match **`extract_interpret_data.py`** as long as the **on-disk layout** matches this document.

## Acceptance criteria

- [ ] **No** episode-root-only `/cameras/` as the sole image location; each timestep has **`/timestep_*/cameras/*`** populated.
- [ ] `token_meta` JSON parses to **867** entries; image patches use **`"image_patch"`**.
- [ ] All **`attention`** datasets: `float32`, shape **`(8, 51, 867)`** for every sampled layer.
- [ ] All **`tsne`** datasets: **`(867, 2)`** `float32`.
- [ ] All **`neighbors`** datasets: compound **`(50, 5)`** with **`<i4` / `<f4`** fields.
- [ ] **`meta.attrs`** exactly as specified.
- [ ] `h5py.File(path, "r")` from a **fresh process** reads all groups without errors.
