# Epic: Backend Development (Track C)

FastAPI service that reads pre-computed Pi-Zero interpretability HDF5 files and exposes a structured, read-only REST API for the React frontend. No GPU and no extraction logic—pure I/O, validation, and JSON/PNG responses.

## Architecture

```
HDF5 files (data/*.h5)
       │
       ▼
EpisodeIndex  ──scans directory on startup──►  dict[episode_id → Path]
       │
       ▼
HDF5Reader   (Repository pattern: all h5py access lives here)
       │
       ▼
Routers      (episodes, attention, embedding)  ──►  Pydantic models → JSON / PNG
```

- **HDF5Reader** encapsulates every dataset read, timestep key resolution, and camera detection. HTTP layers never import `h5py` directly.
- **Routers** validate query parameters, map HTTP errors (404/422), and call the reader.
- **Schemas** (`Pydantic v2`) mirror Contract 2 so the frontend can share types.

## Zero ML Dependencies

The backend must **not** depend on sklearn, scipy, JAX, or PyTorch. t-SNE coordinates and neighbor indices are **pre-computed in Track B**; Track C only loads arrays and serves them.

Allowed core stack: **FastAPI**, **uvicorn**, **h5py**, **numpy**, **Pillow** (PNG encoding), **pydantic** v2 (+ **pydantic-settings** for config).

## Task Table

| Task | Focus | Prompt |
|------|--------|--------|
| **C1** | App scaffold, lifespan, CORS, health, dependencies | [C1-fastapi-scaffold/prompt.md](C1-fastapi-scaffold/prompt.md) |
| **C2** | `EpisodeIndex`, `HDF5Reader`, constants, Pydantic schemas, pytest fixture | [C2-hdf5-data-layer/prompt.md](C2-hdf5-data-layer/prompt.md) |
| **C3** | Episode list/detail, camera PNG, token-meta routes | [C3-episode-metadata-api/prompt.md](C3-episode-metadata-api/prompt.md) |
| **C4** | Attention row + breakdown, attention summary | [C4-attention-api/prompt.md](C4-attention-api/prompt.md) |
| **C5** | t-SNE scatter + neighbors | [C5-embedding-api/prompt.md](C5-embedding-api/prompt.md) |

## Known Pitfalls (Backend-Relevant)

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **#8 Timestep key mismatch** | Frontend sends sequential `t ∈ {0,1,…}` but HDF5 groups are `timestep_000`, `timestep_010`, … | Sort all `timestep_*` keys lexicographically and map index `i` → `keys[i]` via `_resolve_ts_key()`. |
| **#10 Camera detection** | `get_meta()` needs `camera_names` but per-episode `/cameras/` may be absent | `_detect_cameras()`: prefer episode-level `/cameras/` if present; else inspect **first** timestep’s `timestep_NNN/cameras/`. |
| **CORS** | Browser blocks API when UI is opened from another host on the LAN | Use `CORSMiddleware` with `allow_origins=["*"]` for development/LAN testing (see C1). |
| **Camera timestep** | Wrong frame shown in the UI | `GET .../camera/{cam}` must accept `?timestep=N` and pass **sequential** `N` into `get_camera_image(..., timestep=N)` so the reader resolves the correct HDF5 group. |

## REST API (Contract 2)

| Method & path | Response | Shape / notes |
|---------------|----------|----------------|
| `GET /api/health` | JSON | `{"status": "ok"}` |
| `GET /api/episodes` | JSON | `list[{episode_id, task_instruction, num_timesteps}]` → `EpisodeSummary` |
| `GET /api/episodes/{id}` | JSON | `{episode_id, task_instruction, num_timesteps, instruction_tokens, sampled_layers, camera_names}` → `EpisodeMeta` |
| `GET /api/episodes/{id}/camera/{cam}?timestep=N` | `image/png` | Optional `timestep`; reader uses per-timestep path when present, else episode-level fallback |
| `GET /api/episodes/{id}/timesteps/{t}/token-meta` | JSON | `list[{index, type, source, patch_row?, patch_col?, token_text?, token_position?}]` → `TokenMeta` (`t` sequential) |
| `GET /api/episodes/{id}/timesteps/{t}/attention?layer=L&head=H&action=A` | JSON | `{row: float[867], breakdown: {...}}` → `AttentionResponse` |
| `GET /api/episodes/{id}/timesteps/{t}/attention/summary?layer=L&head=H` | JSON | `{modality_totals: dict[str,float], per_action: float[50]}` → `AttentionSummary` |
| `GET /api/episodes/{id}/timesteps/{t}/tsne?layer=L` | JSON | `{points: TsnePoint[867]}` → `TsneResponse` |
| `GET /api/episodes/{id}/timesteps/{t}/tsne/neighbors?layer=L&action=A` | JSON | `{selected: {index,x,y}, neighbors: NearestNeighbor[5]}` → `NeighborResponse` |

**AttentionResponse.breakdown** (`AttentionBreakdownDetail`): `cameras` (per-camera patch weight lists), `camera_totals`, `language_weights`, `language_total`, `state_weight`, `action_weights`, `action_total`.

**TsnePoint**: `index`, `x`, `y`, `type`, `source`, `color`.

**NearestNeighbor**: `index`, `x`, `y`, `distance`, `modality_group`, `type`, `source`.

## Tech Stack

| Package | Role |
|---------|------|
| `fastapi` | HTTP API, dependency injection, OpenAPI |
| `uvicorn` | ASGI server |
| `h5py` | HDF5 read |
| `numpy` | Array handling |
| `Pillow` | RGB array → PNG |
| `pydantic` v2 | Request/response models |
| `pydantic-settings` | `DATA_DIR`, etc. |

## File Placement (implementation)

All runnable code lives under the **`openpi_interpret/backend/`** package (not under this `docs/` tree):

```
openpi_interpret/backend/
  app/
    main.py              # FastAPI app, CORS, router include, lifespan
    config.py            # Settings (e.g. data directory)
    data/
      constants.py       # SAMPLED_LAYERS, TOKEN_RANGES, TOKEN_COLORS, MODALITY_GROUPS, NEIGHBOR_DTYPE, CAMERA_NAMES
      schemas.py         # Pydantic models
      hdf5_reader.py     # EpisodeIndex, HDF5Reader
      dependencies.py    # get_episode_index()
    routers/
      episodes.py
      attention.py
      embedding.py
  requirements.txt
  tests/
    conftest.py          # Test HDF5 fixture generator + TestClient fixtures
    test_episodes.py
    test_attention.py
    test_embedding.py
```

This document directory (`docs/solution_design_v3/backend-development/`) holds **task prompts only**; implementers copy patterns into `backend/` as specified in C1–C5.

## Related

- Parent plan: [../README.md](../README.md) (HDF5 schema, token map, issues #1–#12)
- Engineering rules: `.cursor/rules/engineering-standards.mdc` (Repository pattern, `data/` vs `routers/`)
