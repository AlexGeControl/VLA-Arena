# C3 — Episode & Metadata API

## Goal

Implement the **`episodes`** router and wire it in `main.py`. Endpoints list episodes, return full metadata, serve camera PNGs, and return token metadata for a sequential timestep.

## Dependencies

- `get_episode_index()` in `app/data/dependencies.py`: builds **`EpisodeIndex(settings.data_dir)`** once (app state or lru singleton) and yields it to routes via `Depends()`.
- Reuse **`HDF5Reader`** only through the index; routers do not open HDF5 directly.

## Endpoints

### `GET /api/episodes`

- **Response:** `list[EpisodeSummary]`
- For each id from `index.list_ids()`, open reader, `get_meta()`, append `EpisodeSummary(episode_id=..., task_instruction=..., num_timesteps=...)`.
- Sort order: same as `list_ids()` (sorted episode ids).

### `GET /api/episodes/{episode_id}`

- **Response:** `EpisodeMeta`
- **404** if episode not in index (`KeyError` from `get_reader`).
- Spread `get_meta()` into `EpisodeMeta` (ensure types match schema).

### `GET /api/episodes/{episode_id}/camera/{camera_name}`

- **Response:** `image/png` (`Response` with `media_type="image/png"`).
- **Query:** optional **`timestep`** (`int | None`). When the client sends **`?timestep=N`**, pass **`timestep=N`** into `reader.get_camera_image(camera_name, timestep=N)` so the reader resolves the correct HDF5 group via `_resolve_ts_key`.
- Reader behavior (C2): try per-timestep path first when `timestep` is set; **fall back** to episode-level `/cameras/{name}` if per-timestep dataset missing (legacy files).
- **404** for unknown episode, unknown camera (`KeyError`), or invalid timestep (`ValueError` from `_resolve_ts_key` → treat as not found).
- Encode RGB `uint8` `(H,W,3)` with **Pillow** → PNG bytes (e.g. `io.BytesIO`).

### `GET /api/episodes/{episode_id}/timesteps/{timestep}/token-meta`

- **Response:** `list[TokenMeta]`
- Path parameter **`timestep`** is the **sequential** index (0-based); reader uses `_resolve_ts_key`.
- **404** if episode missing or `ValueError` from reader (timestep OOR).
- Map each dict from `get_token_meta` to `TokenMeta(**d)` (ensure JSON keys match schema).

## Router Module

- `APIRouter(prefix="/api/episodes", tags=["episodes"])`
- Import schemas from `app.data.schemas`, index from `app.data.hdf5_reader`, `Depends(get_episode_index)`.

## `main.py`

- `app.include_router(...)` for the episodes router.

## Acceptance Criteria

1. `GET /api/episodes` returns 200 and a JSON array; each item has `episode_id`, `task_instruction`, `num_timesteps`.
2. `GET /api/episodes/{valid_id}` returns 200 and includes `instruction_tokens`, `sampled_layers`, `camera_names`.
3. `GET /api/episodes/{valid_id}/camera/{valid_cam}?timestep=0` returns `Content-Type: image/png` and non-empty body when fixture has per-timestep cameras.
4. Query param **`timestep`** is forwarded to **`get_camera_image`** (verified by a test with two timesteps and different image contents, or mock/spy if used).
5. `GET .../timesteps/0/token-meta` returns 867 elements (for standard contract fixture) with valid JSON schema.
6. Unknown episode → **404**; bad timestep → **404** (or consistent documented status).
7. Tests in `tests/test_episodes.py` use `TestClient` and the C2 fixture path.
