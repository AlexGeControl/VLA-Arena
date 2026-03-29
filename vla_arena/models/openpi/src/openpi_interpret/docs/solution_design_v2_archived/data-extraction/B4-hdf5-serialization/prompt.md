# Task B4: Serialization to HDF5

> Part of the [Data Extraction Pipeline](../README.md) epic. Depends on [B2](../B2-attention-capture/prompt.md) and [B3](../B3-tsne-neighbors/prompt.md).

## Goal

Write captured attention weights, pre-computed t-SNE coordinates, pre-computed nearest neighbors, Q-projections, and episode metadata to HDF5 files that the FastAPI backend can efficiently read.

## Task

Create `openpi_interpret/extraction/serialize.py`.

### HDF5 File Layout

One file per episode:

```
{output_dir}/{episode_id}.h5
  /meta
    attrs:
      episode_id        (str)
      task_instruction   (str)
      num_timesteps      (int)
      instruction_tokens (str, JSON array)
      sampled_layers     (str, JSON array)
  /cameras
    /base_0_rgb          # dataset: uint8 [H, W, 3]
    /left_wrist_0_rgb    # dataset: uint8 [H, W, 3]
    (right_wrist_0_rgb omitted if masked)
  /timestep_000
    /token_meta          # dataset: variable-length string (JSON array of 867 entries)
    /attention
      /layer_00          # dataset: float32 [8, 51, 867], chunks=(1, 51, 867)
      /layer_03
      /layer_06
      /layer_09
      /layer_12
      /layer_15
      /layer_17
    /tsne
      /layer_00          # dataset: float32 [867, 2]
      /layer_03  ...
    /neighbors
      /layer_00          # dataset: compound [50, 5]
                         #   fields: neighbor_index (int32), distance (float32)
      /layer_03  ...
    /q_projections
      /layer_00
        /prefix          # dataset: float32 [816, 8, 256]
        /suffix          # dataset: float32 [51, 8, 256]
      /layer_03  ...
  /timestep_001
    ...
```

### Writer Implementation

```python
import h5py
import json
import numpy as np
from pathlib import Path
from PIL import Image

NEIGHBOR_DTYPE = np.dtype([("neighbor_index", np.int32), ("distance", np.float32)])

def write_episode_hdf5(
    output_dir: Path,
    episode_id: str,
    task_instruction: str,
    instruction_tokens: list[str],
    sampled_layers: list[int],
    camera_images: dict[str, np.ndarray | None],
    timesteps: list[dict],
):
    """Write a complete episode to HDF5.

    Args:
        camera_images: {camera_name: uint8 [H,W,3] or None if masked}
        timesteps: list of dicts, each with keys:
            - timestep_index: int
            - token_meta: list[dict] (867 entries)
            - attention: {layer: ndarray [8, 51, 867]}
            - tsne_coords: {layer: ndarray [867, 2]}
            - neighbors: {layer: ndarray [50, 5] structured}
            - q_projections: {layer: (prefix [816,8,256], suffix [51,8,256])}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{episode_id}.h5"

    with h5py.File(h5_path, "w") as f:
        # Meta
        meta = f.create_group("meta")
        meta.attrs["episode_id"] = episode_id
        meta.attrs["task_instruction"] = task_instruction
        meta.attrs["num_timesteps"] = len(timesteps)
        meta.attrs["instruction_tokens"] = json.dumps(instruction_tokens)
        meta.attrs["sampled_layers"] = json.dumps(sampled_layers)

        # Cameras
        cam_grp = f.create_group("cameras")
        for cam_name, img in camera_images.items():
            if img is not None:
                cam_grp.create_dataset(cam_name, data=img, dtype=np.uint8)

        # Timesteps
        for ts in timesteps:
            ts_grp = f.create_group(f"timestep_{ts['timestep_index']:03d}")

            # Token metadata as JSON string
            ts_grp.create_dataset(
                "token_meta",
                data=json.dumps(ts["token_meta"]),
                dtype=h5py.string_dtype(),
            )

            # Attention
            attn_grp = ts_grp.create_group("attention")
            for layer, attn in ts["attention"].items():
                attn_grp.create_dataset(
                    f"layer_{layer:02d}",
                    data=attn,
                    dtype=np.float32,
                    chunks=(1, 51, 867),
                )

            # t-SNE
            tsne_grp = ts_grp.create_group("tsne")
            for layer, coords in ts["tsne_coords"].items():
                tsne_grp.create_dataset(f"layer_{layer:02d}", data=coords, dtype=np.float32)

            # Neighbors
            nbr_grp = ts_grp.create_group("neighbors")
            for layer, nbrs in ts["neighbors"].items():
                nbr_grp.create_dataset(f"layer_{layer:02d}", data=nbrs)

            # Q-projections (backlog asset)
            qproj_grp = ts_grp.create_group("q_projections")
            for layer, (q_pre, q_suf) in ts["q_projections"].items():
                layer_grp = qproj_grp.create_group(f"layer_{layer:02d}")
                layer_grp.create_dataset("prefix", data=np.asarray(q_pre), dtype=np.float32)
                layer_grp.create_dataset("suffix", data=np.asarray(q_suf), dtype=np.float32)


def write_episodes_index(output_dir: Path, episodes: list[dict]):
    """Write episodes-index.json (still needed for quick listing without opening HDF5)."""
    index_path = output_dir / "episodes-index.json"
    with open(index_path, "w") as f:
        json.dump(episodes, f, indent=2)
```

### Attention Storage: Float32, Chunked

Attention is stored as **float32** (not uint8-quantized). The backend performs slicing directly on the HDF5 dataset. Chunks of `(1, 51, 867)` allow reading a single head's attention without loading all 8 heads.

### Output Files

```
openpi_interpret/extraction/serialize.py
```

## Acceptance Criteria

- [ ] Each episode produces a single `.h5` file
- [ ] `/meta` group has all required attributes
- [ ] `/cameras` has datasets for non-masked cameras
- [ ] Attention datasets have shape `[8, 51, 867]` and chunks `(1, 51, 867)`
- [ ] t-SNE datasets have shape `[867, 2]`
- [ ] Neighbor datasets have shape `[50, 5]` with compound dtype
- [ ] Q-projection datasets have correct shapes `[816, 8, 256]` and `[51, 8, 256]`
- [ ] Token metadata stored as valid JSON string
- [ ] `episodes-index.json` lists all processed episodes
- [ ] HDF5 files can be opened and read by `h5py` in a separate process
