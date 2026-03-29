# Task B3: t-SNE & Nearest-Neighbor Pre-Computation

> Part of the [Data Extraction Pipeline](../README.md) epic. Depends on [B2](../B2-attention-capture/prompt.md).

## Goal

Project all tokens into the shared attention head space using Q-projection matrices, compute 2D t-SNE coordinates per sampled layer, and pre-compute nearest neighbors per action token per modality group. All results are stored in HDF5 for lightweight backend serving.

## Background: Head-Space Projection

Pi-Zero's two experts have different hidden dimensions (prefix: 2048, suffix: 1024). We project all tokens into the **shared attention head space** using Q-projection matrices:

1. Q projections map both experts into head space: `[B, seq, 8, 256]`
2. Reshape to `[seq, 8*256] = [seq, 2048]` (concatenate heads)
3. Concatenate prefix and suffix: `[867, 2048]`
4. Run t-SNE → `[867, 2]`

**Why head space?** This is the space where cross-expert attention operates. Tokens that produce similar queries seek similar information and are meaningfully close.

## Task

Create `openpi_interpret/extraction/tsne.py`:

### t-SNE Computation

```python
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial import KDTree

SAMPLED_LAYERS = [0, 3, 6, 9, 12, 15, 17]
MODALITY_GROUPS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "language", "state"]
MODALITY_RANGES = {
    "base_0_rgb":        (0, 256),
    "left_wrist_0_rgb":  (256, 512),
    "right_wrist_0_rgb": (512, 768),
    "language":          (768, 816),
    "state":             (816, 817),
}
ACTION_RANGE = (817, 867)

def compute_head_space_tsne(
    q_projections: dict[int, list],
    sampled_layers: list[int] = SAMPLED_LAYERS,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> dict[int, np.ndarray]:
    """Compute t-SNE from Q-projected representations.

    Returns:
        {layer_idx: ndarray[867, 2]} — t-SNE coordinates per layer.
    """
    results = {}

    for layer in sampled_layers:
        q_prefix, q_suffix = q_projections[layer]
        prefix_flat = np.asarray(q_prefix[0]).reshape(q_prefix.shape[1], -1)  # [816, 2048]
        suffix_flat = np.asarray(q_suffix[0]).reshape(q_suffix.shape[1], -1)  # [51, 2048]
        all_tokens = np.concatenate([prefix_flat, suffix_flat], axis=0)        # [867, 2048]

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        coords = tsne.fit_transform(all_tokens)  # [867, 2]
        results[layer] = coords

    return results
```

### Nearest-Neighbor Pre-Computation

For each action token (50 total) at each sampled layer, find the single nearest neighbor in each of the 5 modality groups by Euclidean distance in t-SNE space:

```python
def compute_nearest_neighbors(
    tsne_coords: dict[int, np.ndarray],
    sampled_layers: list[int] = SAMPLED_LAYERS,
) -> dict[int, np.ndarray]:
    """Pre-compute nearest neighbors per action token per modality group.

    Returns:
        {layer_idx: structured ndarray [50, 5]}
        Each entry has fields: neighbor_index (int32), distance (float32)
        Axis 0 = action token (0-49), axis 1 = modality group index (see MODALITY_GROUPS)
    """
    dtype = np.dtype([("neighbor_index", np.int32), ("distance", np.float32)])
    results = {}

    for layer in sampled_layers:
        coords = tsne_coords[layer]  # [867, 2]
        neighbors = np.zeros((50, 5), dtype=dtype)

        for group_idx, group_name in enumerate(MODALITY_GROUPS):
            start, end = MODALITY_RANGES[group_name]
            group_coords = coords[start:end]  # [N, 2]
            group_indices = np.arange(start, end)

            tree = KDTree(group_coords)

            for action_idx in range(50):
                action_token_idx = 817 + action_idx
                action_coord = coords[action_token_idx]
                dist, local_idx = tree.query(action_coord)
                neighbors[action_idx, group_idx] = (group_indices[local_idx], dist)

        results[layer] = neighbors

    return results
```

### Output

Both `tsne_coords` and `neighbors` are passed to Task B4 (Serialization) for storage in HDF5:

- `tsne_coords[layer]` → `/timestep_{t}/tsne/layer_{ll}` as `float32 [867, 2]`
- `neighbors[layer]` → `/timestep_{t}/neighbors/layer_{ll}` as structured `[50, 5]`
- Q-projections → `/timestep_{t}/q_projections/layer_{ll}/{prefix,suffix}` (backlog asset)

### Output File

```
openpi_interpret/extraction/tsne.py
```

## Acceptance Criteria

- [ ] t-SNE produces `[867, 2]` float32 coordinates per sampled layer
- [ ] Coordinates are finite (no NaN or Inf)
- [ ] Per-layer t-SNE takes < 30 seconds (867 points in 2048 dims)
- [ ] Different layers produce different layouts (not identical)
- [ ] Neighbors array has shape `[50, 5]` per layer with valid indices and distances
- [ ] Each neighbor index falls within the correct modality range
- [ ] Distances are non-negative
