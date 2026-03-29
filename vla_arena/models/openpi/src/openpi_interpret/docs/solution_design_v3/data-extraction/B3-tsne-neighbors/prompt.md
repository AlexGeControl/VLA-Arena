# Task B3: t-SNE + nearest neighbors (head-space Q)

> Part of the [Data Extraction epic](../README.md). Depends on [B2](../B2-attention-capture/prompt.md).

## Goal

From **Q-projections** captured per layer:

1. Build a single **(867, 2048)** matrix per layer by concatenating prefix and suffix head features.
2. Run **t-SNE** to 2D per layer.
3. For **each of 50 action tokens** (suffix positions aligned with the UI), find the **nearest neighbor** in each of **five modality groups** using **scipy** `KDTree`.

## Q-projection layout

Per sampled layer:

- **Prefix**: `(816, 8, 256)` → reshape to **`(816, 2048)`** (flatten heads: `8 * 256`).
- **Suffix**: `(51, 8, 256)` → reshape to **`(51, 2048)`**.
- **Concatenate** along token axis: **`(867, 2048)`** = `[prefix_rows; suffix_rows]`.

```python
def head_space_matrix(q_prefix: np.ndarray, q_suffix: np.ndarray) -> np.ndarray:
    # q_prefix: (816, 8, 256), q_suffix: (51, 8, 256)
    p = q_prefix.reshape(816, 8 * 256)
    s = q_suffix.reshape(51, 8 * 256)
    return np.concatenate([p, s], axis=0)  # (867, 2048)
```

## t-SNE

Use **sklearn**:

```python
from sklearn.manifold import TSNE

coords = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
).fit_transform(features.astype(np.float64))  # (867, 2)
```

Store **`float32`** in HDF5. Ensure **no NaN/Inf** before writing.

## Modality groups (key ranges along the 867 axis)

| Group | Index range (inclusive) | Notes |
|-------|-------------------------|--------|
| `base_0_rgb` | **0 – 255** | Image patches, camera base |
| `left_wrist_0_rgb` | **256 – 511** | |
| `right_wrist_0_rgb` | **512 – 767** | May be masked in images; indices still exist in token sequence |
| `language` | **768 – 815** | |
| `state` | **816** | Single token index (state slot); tree still works with one point |

**Action tokens** for neighbor queries: **50** positions with **global indices `817–866`** (suffix `action` tokens in the 867-length sequence; see Token Map in [solution_design_v3 README](../README.md)). Encode this as a constant, e.g. `ACTION_TOKEN_INDICES = list(range(817, 867))`, and assert alignment with `build_token_meta()`.

## Nearest neighbors (`scipy.spatial.KDTree`)

For each sampled layer and each modality group:

1. Build **`KDTree`** on **`features[group_indices]`**.
2. For each action token row `a`, query `tree.query(a, k=1)` (or `k=2` and skip self if ever in the same set).
3. Store **global** neighbor index in **`0..866`**.

**Output dtype** (per HDF5 compound array):

```python
import numpy as np

NEIGHBOR_DTYPE = np.dtype([
    ("neighbor_index", "<i4"),
    ("distance", "<f4"),
])
```

Final **`neighbors[layer]`** shape: **`(50, 5)`** — 50 actions × 5 modality groups.

## File placement

Implement in **`openpi_interpret/extraction/tsne.py`**:

- `compute_tsne_per_layer(q_prefix_by_layer, q_suffix_by_layer, sampled_layers) -> dict[int, np.ndarray]`
- `compute_neighbors_per_layer(features_by_layer, action_token_indices) -> dict[int, np.ndarray]`

Keep **constants** (ranges, 50 actions, 867) in a small **`constants.py`** or module-level names per engineering standards.

## Acceptance criteria

- [ ] Per layer: concatenated feature shape **`(867, 2048)`** before t-SNE.
- [ ] t-SNE: `n_components=2`, `perplexity=30`, `random_state=42`.
- [ ] Output maps: `layer -> (867, 2)` `float32`, all finite.
- [ ] Neighbors: `layer -> (50, 5)` structured, dtype **`neighbor_index` `<i4`**, **`distance` `<f4`**.
- [ ] Every `neighbor_index` in **`[0, 866]`** and refers to a token in the **correct** modality group’s index set.
- [ ] Unit test (CPU): run t-SNE + trees on **synthetic** `(867, 2048)` data to validate shapes and index ranges without a GPU.
