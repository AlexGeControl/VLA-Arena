# C2 — HDF5 Data Layer

## Goal

Implement **`EpisodeIndex`**, **`HDF5Reader`**, shared **constants**, all **Pydantic schemas** for Contract 2, and a **pytest** HDF5 fixture generator in `conftest.py`. No FastAPI imports in the data layer.

## Pitfall #8: Timestep Key Resolution

HDF5 timestep groups use **raw frame indices** in the name (`timestep_000`, `timestep_010`, …), not `timestep_0`, `timestep_1`. The API and frontend use **sequential** indices `0, 1, 2, …`.

**Implement:**

- `_get_timestep_keys()` — open file, collect every top-level key starting with `timestep_`, **sort** lexicographically (sorting `timestep_*` strings matches numeric frame order), cache on the reader instance.
- `_resolve_ts_key(timestep: int) -> str` — return `keys[timestep]` or raise **`ValueError`** with a clear message if out of range.

All per-timestep reads (`token_meta`, `attention`, `tsne`, `neighbors`, per-timestep cameras) must go through `_resolve_ts_key`.

## Pitfall #10: Camera Detection

After cameras moved **per-timestep**, episode-level `/cameras/` may be **absent**. `get_meta()` still needs `camera_names`.

**Implement `_detect_cameras(f: h5py.File) -> list[str]`:**

1. If top-level group `"cameras"` exists, return names from a fixed allowlist (see constants) that exist under `f["cameras"]`.
2. Else, find sorted `timestep_*` keys; if non-empty and `f[first_ts]["cameras"]` exists, return allowlist names present there.
3. Else return `[]`.

## `EpisodeIndex`

- **`__init__(data_dir)`**: resolve `Path`, scan `*.h5` files, map **`stem` → path** (episode id = filename without `.h5`).
- **`list_ids() -> list[str]`**: sorted episode ids.
- **`get_reader(episode_id) -> HDF5Reader`**: raise **`KeyError`** if unknown.
- **`__contains__(episode_id)`** for convenience.

Scan once at construction (or lazy on first list); no need to watch filesystem changes for v1.

## `HDF5Reader`

Constructor takes `Path` to one `.h5` file.

**Methods:**

| Method | Behavior |
|--------|----------|
| `get_meta()` | Read `/meta` attrs: `episode_id`, `task_instruction`, `num_timesteps`, `instruction_tokens` (JSON string → list), `sampled_layers` (JSON string → list). Add **`camera_names`** via `_detect_cameras`. Return a dict suitable for `EpisodeMeta(**meta)` after typing coercion. |
| `get_camera_image(camera_name, timestep=None)` | If `timestep` is not `None`, resolve ts key, read `/{ts}/cameras/{name}` if present. Else or if missing, fall back to `/cameras/{name}`. Return **`uint8` ndarray shape `(H,W,3)`**. Raise **`KeyError`** if camera missing at all levels. |
| `get_token_meta(timestep)` | JSON string at `/{ts}/token_meta` → **`list[dict]`** (867 entries). |
| `get_attention(timestep, layer)` | Dataset `/{ts}/attention/layer_{layer:02d}`, shape `(8, 51, 867)`, `float32`. Validate layer ∈ sampled set. |
| `get_tsne(timestep, layer)` | `/{ts}/tsne/layer_{layer:02d}`, shape `(867, 2)`, `float32`. |
| `get_neighbors(timestep, layer)` | `/{ts}/neighbors/layer_{layer:02d}`, structured array shape `(50, 5)` with fields below. |
| `get_sampled_layers()` | From meta attrs (optional helper). |

Open file per method (or minimal scope with context manager); rely on OS mmap.

**Layer validation:** if `layer not in SAMPLED_LAYERS`, raise **`ValueError`** with message listing allowed layers.

## Constants (`app/data/constants.py`)

Define exactly (adjust only if Contract 1 changes):

```python
SAMPLED_LAYERS = [0, 3, 6, 9, 12, 15, 17]

CAMERA_NAMES = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

TOKEN_RANGES: dict[str, tuple[int, int]] = {
    "base_0_rgb": (0, 256),
    "left_wrist_0_rgb": (256, 512),
    "right_wrist_0_rgb": (512, 768),
    "language": (768, 816),
    "state": (816, 817),
    "action": (817, 867),
}

TOKEN_COLORS: dict[str, str] = {
    "base_0_rgb": "#3B82F6",
    "left_wrist_0_rgb": "#06B6D4",
    "right_wrist_0_rgb": "#14B8A6",
    "language": "#F97316",
    "state": "#22C55E",
    "action": "#EF4444",
}

MODALITY_GROUPS = [
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "language",
    "state",
]

NEIGHBOR_DTYPE = [("neighbor_index", "<i4"), ("distance", "<f4")]
```

## Pydantic Schemas (`app/data/schemas.py`)

Mirror the backend README / parent design:

- `EpisodeSummary` — `episode_id`, `task_instruction`, `num_timesteps`
- `EpisodeMeta` — adds `instruction_tokens`, `sampled_layers`, `camera_names`
- `TokenMeta` — `index`, `type`, `source`, optional `patch_row`, `patch_col`, `token_text`, `token_position`
- `AttentionBreakdownDetail` — `cameras`, `camera_totals`, `language_weights`, `language_total`, `state_weight`, `action_weights`, `action_total`
- `AttentionResponse` — `row`, `breakdown`
- `AttentionSummary` — `modality_totals`, `per_action`
- `TsnePoint` — `index`, `x`, `y`, `type`, `source`, `color`
- `TsneResponse` — `points`
- `SelectedPoint` — `index`, `x`, `y`
- `NearestNeighbor` — `index`, `x`, `y`, `distance`, `modality_group`, `type`, `source`
- `NeighborResponse` — `selected`, `neighbors`

No `Any` in public model fields; use precise types.

## Test HDF5 Fixture (`tests/conftest.py`)

Implement a **generator** that writes a **minimal valid** `.h5` file to a `tmp_path`:

- `/meta` attrs: string/int/JSON strings as in Contract 1 (`sampled_layers` JSON list, `instruction_tokens` JSON list).
- At least **two** timestep groups with **non-sequential suffixes** (e.g. `timestep_000`, `timestep_020`) to test **`_resolve_ts_key`** maps `0 → timestep_000`, `1 → timestep_020`.
- Per-timestep: `token_meta` (JSON list length 867 minimal stubs), `attention/layer_00` only if you use layer `0` in tests—or use a layer from `SAMPLED_LAYERS` and name dataset `layer_XX` accordingly.
- `tsne/layer_XX` shape `(867,2)`, `neighbors/layer_XX` compound dtype `NEIGHBOR_DTYPE` shape `(50,5)`.
- Cameras **only** under first timestep’s `cameras/` (no episode-level) to test **pitfall #10**.
- Fixture exposes path and/or `EpisodeIndex` + `DATA_DIR` for `TestClient` (wired in C3+).

Use `h5py` in the fixture only; tests for C2 can assert reader methods return expected shapes and that timestep index `1` reads the second group name.

## Acceptance Criteria

1. `_resolve_ts_key(0)` and `_resolve_ts_key(1)` return distinct HDF5 group names matching sorted `timestep_*` order.
2. `_detect_cameras` returns the three camera names when they exist only under `timestep_*/cameras/`.
3. `get_meta()` includes `camera_names` consistent with the file.
4. `get_attention`, `get_tsne`, `get_neighbors` validate `layer` against `SAMPLED_LAYERS`.
5. All listed Pydantic models import cleanly and match field names in the parent [README.md](../README.md) API table.
6. `conftest.py` builds a file on the fly; pytest can run without real extraction output.
7. **No** `sklearn`, `scipy`, or `jax` imports anywhere in `app/data/`.
