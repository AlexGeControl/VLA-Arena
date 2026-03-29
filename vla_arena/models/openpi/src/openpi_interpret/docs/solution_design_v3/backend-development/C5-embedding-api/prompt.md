# C5 — Embedding API (t-SNE & Neighbors)

## Goal

Serve pre-computed **2D t-SNE coordinates** for all 867 tokens and **5 nearest neighbors** per modality group for a selected action token. Coloring rules must match the frontend legend.

## Router

- Prefix: **`/api/episodes/{episode_id}/timesteps/{timestep}/tsne`**
- Tag: e.g. `embedding`

## Token Coloring

When building **`TsnePoint.color`**:

- If `meta["type"] == "image_patch"`: **`color_key = meta["source"]`** (camera/stream id, e.g. `base_0_rgb`).
- Else: **`color_key = meta["type"]`** (e.g. `language`, `state`, `action`).
- **`color = TOKEN_COLORS.get(color_key, "#888888")`** fallback for unexpected keys.

## `MODALITY_GROUPS`

Neighbors are stored with one row per action and **5** slots aligned to modality groups:

```python
MODALITY_GROUPS = [
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "language",
    "state",
]
```

The HDF5 dataset `neighbors/layer_XX` has shape **`(50, 5)`**: for action index `a`, column `m` corresponds to `MODALITY_GROUPS[m]`.

## `GET ""` — t-SNE scatter

**Query:** `layer` (required, int).

**Validation:** `layer in SAMPLED_LAYERS`; else **422**.

**Response:** `TsneResponse` with **`points`** length **867**.

**Steps:**

1. `coords = reader.get_tsne(timestep, layer)` → `(867, 2)`.
2. `token_metas = reader.get_token_meta(timestep)` → 867 dicts.
3. For each `i`, build `TsnePoint(index=i, x=coords[i,0], y=coords[i,1], type=..., source=..., color=...)` using coloring rules above.

**404** if episode missing; **404** if timestep OOR; handle missing dataset consistently.

## `GET "/neighbors"`

**Query:** `layer`, `action` (required).

**Validation:**

- `layer in SAMPLED_LAYERS`
- `0 <= action < 50`

**Response:** `NeighborResponse`:

- **Global index** of the action token: `action_global = TOKEN_RANGES["action"][0] + action` (i.e. **817 + action**).
- **`selected`:** `SelectedPoint(index=action_global, x=coords[action_global,0], y=coords[action_global,1])`.
- **`neighbors`:** list of **5** `NearestNeighbor` objects:
  - For `mod_idx, mod_name in enumerate(MODALITY_GROUPS)`: read `entry = nbr_data[action, mod_idx]` with fields `neighbor_index` (int), `distance` (float).
  - `n_idx = entry["neighbor_index"]`; populate `x,y` from `coords[n_idx]`, `modality_group=mod_name`, `type`/`source` from `token_metas[n_idx]`.

## Files

- `app/routers/embedding.py` — router, `_validate_layer`, `_color_for_meta`, `_build_points`, `_build_neighbors`.
- Include router in `main.py`.

## Acceptance Criteria

1. `GET .../tsne?layer=0` (or another valid sampled layer present in fixture) returns **867** points with `x`, `y`, `color` present.
2. Image-patch points use **source**-based colors; non-patch types use **type**-based colors (assert in tests with crafted `token_meta`).
3. `GET .../tsne/neighbors?layer=0&action=0` returns **`selected`** with `index == 817` and exactly **5** neighbors with distinct `modality_group` values matching `MODALITY_GROUPS`.
4. Invalid `layer` or `action` → **422**.
5. **`grep -r sklearn\\|scipy` on `app/`** (or equivalent) shows **no** matches in backend application code.
6. Tests in `tests/test_embedding.py` use `TestClient` + C2 fixture.
