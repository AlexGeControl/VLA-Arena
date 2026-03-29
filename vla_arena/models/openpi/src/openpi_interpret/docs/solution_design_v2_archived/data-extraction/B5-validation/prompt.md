# Task B5: End-to-End Validation

> Part of the [Data Extraction Pipeline](../README.md) epic. Depends on [B4](../B4-hdf5-serialization/prompt.md).

## Goal

Run the full extraction pipeline on a small set of episodes and validate the HDF5 output format, correctness, and performance.

## Task

### 1. Run Extraction

Process 2–3 episodes from L0 S with diverse tasks:

```bash
python extract_interpret_data.py \
  --checkpoint "$CKPT_DIR" \
  --max-episodes 3 \
  --timestep-stride 30 \
  --output-dir ../data
```

### 2. Validate HDF5 Output

Create `openpi_interpret/extraction/validate.py`:

```python
import h5py
import json
import numpy as np
from pathlib import Path

SAMPLED_LAYERS = [0, 3, 6, 9, 12, 15, 17]
NEIGHBOR_DTYPE = np.dtype([("neighbor_index", np.int32), ("distance", np.float32)])

def validate_episode(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        # Meta
        meta = f["/meta"]
        assert "episode_id" in meta.attrs
        assert "task_instruction" in meta.attrs
        num_ts = meta.attrs["num_timesteps"]
        layers = json.loads(meta.attrs["sampled_layers"])
        assert layers == SAMPLED_LAYERS

        # Cameras
        assert "cameras" in f
        assert "base_0_rgb" in f["cameras"]
        img = f["cameras/base_0_rgb"][:]
        assert img.ndim == 3 and img.shape[2] == 3

        for t in range(num_ts):
            ts_key = f"timestep_{t:03d}"
            assert ts_key in f

            # Token metadata
            token_meta = json.loads(f[f"{ts_key}/token_meta"][()])
            assert len(token_meta) == 867

            for layer in SAMPLED_LAYERS:
                layer_key = f"layer_{layer:02d}"

                # Attention
                attn = f[f"{ts_key}/attention/{layer_key}"][:]
                assert attn.shape == (8, 51, 867), f"Attention shape: {attn.shape}"
                assert attn.dtype == np.float32
                row_sums = attn.sum(axis=-1)
                assert np.allclose(row_sums, 1.0, atol=0.01), \
                    f"Row sums: {row_sums.min():.4f}–{row_sums.max():.4f}"

                # t-SNE
                tsne = f[f"{ts_key}/tsne/{layer_key}"][:]
                assert tsne.shape == (867, 2)
                assert np.all(np.isfinite(tsne))

                # Neighbors
                nbrs = f[f"{ts_key}/neighbors/{layer_key}"][:]
                assert nbrs.shape == (50, 5)
                assert nbrs.dtype == NEIGHBOR_DTYPE
                for action_idx in range(50):
                    for group_idx in range(5):
                        ni = nbrs[action_idx, group_idx]["neighbor_index"]
                        dist = nbrs[action_idx, group_idx]["distance"]
                        assert 0 <= ni < 867, f"Invalid neighbor index: {ni}"
                        assert dist >= 0, f"Negative distance: {dist}"

                # Q-projections
                qp = f[f"{ts_key}/q_projections/{layer_key}/prefix"][:]
                qs = f[f"{ts_key}/q_projections/{layer_key}/suffix"][:]
                assert qp.shape == (816, 8, 256)
                assert qs.shape == (51, 8, 256)

    print(f"PASS: {h5_path.name}")
```

### 3. Report Metrics

| Metric | Expected |
|--------|----------|
| Per-episode HDF5 size | ~40–60 MB (float32 attention, uncompressed) |
| Per-timestep extraction time | ~5–15 seconds (inference + t-SNE + neighbors) |
| Peak VRAM | ~7 GB |
| Total for 3 episodes | ~120–180 MB, < 5 minutes |

### 4. Backend Smoke Test

Verify that the backend can serve the extracted data:

```bash
# Terminal 1: start backend
cd openpi_interpret/backend
INTERPRET_DATA_DIR=../data uvicorn app.main:app --port 8080

# Terminal 2: test endpoints
curl http://localhost:8080/api/episodes
curl http://localhost:8080/api/episodes/ep_000/timesteps/0/attention?layer=0\&head=0\&action=0
curl http://localhost:8080/api/episodes/ep_000/timesteps/0/tsne?layer=0
```

### Output File

```
openpi_interpret/extraction/validate.py
```

## Acceptance Criteria

- [ ] All HDF5 validation checks pass for 2–3 episodes
- [ ] Attention row sums are within 1% of 1.0
- [ ] t-SNE coordinates are finite and vary between layers
- [ ] Neighbor indices fall within correct modality ranges
- [ ] Q-projection shapes are correct
- [ ] No CUDA OOM errors during extraction
- [ ] Backend can serve the extracted HDF5 files via REST API
