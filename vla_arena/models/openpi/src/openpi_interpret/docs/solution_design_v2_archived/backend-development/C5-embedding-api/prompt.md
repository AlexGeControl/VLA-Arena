# Task C5: Embedding API

> Part of the [Backend Development](../README.md) epic. Depends on [C2](../C2-hdf5-data-layer/prompt.md).

## Goal

REST endpoints for serving pre-computed t-SNE coordinates and nearest-neighbor results. Pure index reads from HDF5 — no ML computation at runtime.

## Task

### Endpoints

Create `app/routers/embedding.py`:

**`GET /api/episodes/{id}/timesteps/{t}/tsne`** — All 867 t-SNE points for a layer, with metadata and modality colors.

Query parameters:
- `layer` (int, required): One of sampled layers

```python
TOKEN_COLORS = {
    "base_0_rgb": "#3B82F6",
    "left_wrist_0_rgb": "#06B6D4",
    "right_wrist_0_rgb": "#14B8A6",
    "language": "#F97316",
    "state": "#22C55E",
    "action": "#EF4444",
}

@router.get("/api/episodes/{episode_id}/timesteps/{timestep}/tsne",
            response_model=TsneResponse)
async def get_tsne(episode_id: str, timestep: int, layer: int, ...):
    coords = reader.get_tsne(timestep, layer)  # float32 [867, 2]
    token_meta = reader.get_token_meta(timestep)  # list[dict]

    points = []
    for i, (meta, (x, y)) in enumerate(zip(token_meta, coords)):
        color_key = meta["source"] if meta["type"] == "image_patch" else meta["type"]
        points.append(TsnePoint(
            index=i,
            x=float(x),
            y=float(y),
            type=meta["type"],
            source=meta["source"],
            color=TOKEN_COLORS.get(color_key, "#9CA3AF"),
        ))

    return TsneResponse(points=points)
```

**`GET /api/episodes/{id}/timesteps/{t}/tsne/neighbors`** — Nearest neighbors for a selected action token.

Query parameters:
- `layer` (int, required)
- `action` (int, required): `0–49`

The neighbors array in HDF5 is structured as `[50 actions, 5 modality groups]` with fields `(neighbor_index, distance)`. The 5 modality groups are: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`, `language`, `state`.

```python
MODALITY_GROUPS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "language", "state"]

@router.get("/api/episodes/{episode_id}/timesteps/{timestep}/tsne/neighbors",
            response_model=NeighborResponse)
async def get_neighbors(
    episode_id: str, timestep: int, layer: int, action: int, ...
):
    coords = reader.get_tsne(timestep, layer)     # [867, 2]
    neighbors_data = reader.get_neighbors(timestep, layer)  # [50, 5]
    token_meta = reader.get_token_meta(timestep)

    action_token_idx = 817 + action
    selected = {
        "index": action_token_idx,
        "x": float(coords[action_token_idx, 0]),
        "y": float(coords[action_token_idx, 1]),
    }

    neighbors = []
    for group_idx, group_name in enumerate(MODALITY_GROUPS):
        entry = neighbors_data[action, group_idx]
        ni = int(entry["neighbor_index"])
        neighbors.append(NearestNeighbor(
            index=ni,
            x=float(coords[ni, 0]),
            y=float(coords[ni, 1]),
            distance=float(entry["distance"]),
            modality_group=group_name,
            type=token_meta[ni]["type"],
            source=token_meta[ni]["source"],
        ))

    return NeighborResponse(selected=selected, neighbors=neighbors)
```

### Validation

- `layer` must be in `[0, 3, 6, 9, 12, 15, 17]`
- `action` must be in `[0, 49]`

## Acceptance Criteria

- [ ] `GET .../tsne?layer=0` returns 867 points with correct types, sources, and colors
- [ ] `GET .../tsne/neighbors?layer=0&action=0` returns exactly 5 neighbors (one per modality group)
- [ ] Neighbor coordinates match the corresponding points in the t-SNE response
- [ ] Distances are non-negative floats
- [ ] Response time < 50ms (pure HDF5 reads, no computation)
- [ ] No sklearn/scipy imports in the backend codebase
