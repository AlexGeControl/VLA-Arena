# Epic: Backend Development

> **Track C** of the OpenPI InterpreT parallel implementation plan.

This epic builds a FastAPI backend that sits between the data extraction pipeline (Track B) and the frontend SPA (Track A). It reads pre-computed model states from HDF5 files and exposes them through a RESTful API, so the frontend never handles binary data or performs ML computations.

## Architecture

```
┌──────────────────┐         ┌──────────────────────────────────┐         ┌────────────────┐
│  HDF5 Files      │────────►│  FastAPI Backend                 │────────►│  React Frontend│
│  (per episode)   │  h5py   │                                  │  REST   │  (Track A)     │
│                  │  memmap  │  Routers:                        │  JSON   │                │
│  - attention     │         │    /api/episodes                 │         │  Renders:      │
│  - t-SNE coords  │         │    /api/.../attention             │         │  - heatmaps    │
│  - neighbors     │         │    /api/.../tsne                  │         │  - scatter     │
│  - Q-projections │         │    /api/.../camera                │         │  - controls    │
│  - camera images │         │                                  │         │                │
└──────────────────┘         └──────────────────────────────────┘         └────────────────┘
     Track B output                    Track C                                Track A
```

## Design Principles

- **Zero ML dependencies**: The backend has no sklearn, scipy, or JAX. All heavy computation (t-SNE, nearest neighbors) is pre-computed during extraction and stored in HDF5. The backend does pure reads + lightweight numpy slicing.
- **Stateless**: No in-memory caching or session state. Each request reads directly from HDF5 (memory-mapped, so the OS handles caching).
- **Structured responses**: The frontend receives ready-to-render JSON — attention breakdowns, t-SNE points with metadata, neighbors with distances. No client-side binary parsing.

## REST API

```
GET  /api/episodes
     → [{ episode_id, task_instruction, num_timesteps }]

GET  /api/episodes/{id}
     → { episode_id, task_instruction, num_timesteps, instruction_tokens,
         sampled_layers, camera_names }

GET  /api/episodes/{id}/camera/{camera_name}
     → image/png (binary response)

GET  /api/episodes/{id}/timesteps/{t}/token-meta
     → [{ index, type, source, patch_row?, patch_col?, token_text?, token_position? }]

GET  /api/episodes/{id}/timesteps/{t}/attention?layer=0&head=3&action=5
     → { row: number[867],
         breakdown: {
           cameras: { base_0_rgb: number[], left_wrist_0_rgb: number[], right_wrist_0_rgb: number[] },
           camera_totals: { base_0_rgb: number, ... },
           language_weights: number[],
           language_total: number,
           state_weight: number,
           action_weights: number[],
           action_total: number
         }}

GET  /api/episodes/{id}/timesteps/{t}/attention/summary?layer=0&head=3
     → { modality_totals: { images: number, language: number, state: number, actions: number },
         per_action: number[50] }

GET  /api/episodes/{id}/timesteps/{t}/tsne?layer=0
     → { points: [{ index, x, y, type, source, color }...] }

GET  /api/episodes/{id}/timesteps/{t}/tsne/neighbors?layer=0&action=5
     → { selected: { index, x, y },
         neighbors: [{ index, x, y, distance, modality_group, type, source }...] }
```

## HDF5 File Layout (Contract 1: Extraction → Backend)

```
{episode_id}.h5
  /meta
    attrs: episode_id, task_instruction, num_timesteps,
           instruction_tokens (JSON string), sampled_layers (JSON string)
  /cameras
    /base_0_rgb              # dataset: uint8 [H, W, 3]
    /left_wrist_0_rgb        # dataset: uint8 [H, W, 3]
  /timestep_000
    /token_meta              # dataset: variable-length string (JSON)
    /attention
      /layer_00              # dataset: float32 [8, 51, 867], chunked
      /layer_03
      /layer_06
      /layer_09
      /layer_12
      /layer_15
      /layer_17
    /tsne
      /layer_00              # dataset: float32 [867, 2], pre-computed
      /layer_03  ...
    /neighbors
      /layer_00              # dataset: compound [50, 5] — per action × per modality
                             #   fields: neighbor_index (int32), distance (float32)
      /layer_03  ...
    /q_projections           # backlog asset for future re-computation
      /layer_00
        /prefix              # dataset: float32 [816, 8, 256]
        /suffix              # dataset: float32 [51, 8, 256]
  /timestep_001
    ...
```

## Tasks

| Task | Deliverable | Dependency |
|------|------------|------------|
| [C1. FastAPI Scaffold](C1-fastapi-scaffold/prompt.md) | Project setup, uvicorn config, CORS, health check, project structure | None |
| [C2. HDF5 Data Layer](C2-hdf5-data-layer/prompt.md) | Episode index, HDF5 reader class, tensor accessor methods | C1 |
| [C3. Episode & Metadata API](C3-episode-metadata-api/prompt.md) | `/episodes`, `/episodes/{id}`, `/camera`, `/token-meta` endpoints | C2 |
| [C4. Attention API](C4-attention-api/prompt.md) | `/attention` (row + breakdown), `/attention/summary` endpoints | C2 |
| [C5. Embedding API](C5-embedding-api/prompt.md) | `/tsne` (points), `/tsne/neighbors` endpoints (pure HDF5 reads) | C2 |

```
C1 ──► C2 ──► C3
          ├──► C4
          └──► C5
```

C3, C4, C5 are independent of each other and can be developed in parallel after C2.

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | FastAPI | Async, auto-generated OpenAPI docs, Pydantic validation |
| Server | uvicorn | ASGI, production-ready |
| HDF5 access | h5py | Standard Python HDF5 library, memory-mapped reads |
| Array ops | numpy | Slicing attention tensors, reshaping |
| Image serving | Pillow | Encode numpy arrays to PNG for camera endpoints |
| Response models | Pydantic v2 | Typed API responses, auto JSON serialization |

**Notably absent**: No sklearn, scipy, JAX, or torch. The backend is lightweight.

## File Placement

```
openpi_interpret/
  backend/
    app/
      __init__.py
      main.py                 # FastAPI app, CORS, lifespan, uvicorn entry
      config.py               # Settings (data_dir path, host, port)
      routers/
        __init__.py
        episodes.py           # C3: episode list, detail, camera, token-meta
        attention.py          # C4: attention row, breakdown, summary
        embedding.py          # C5: t-SNE points, neighbors
      data/
        __init__.py
        hdf5_reader.py        # C2: HDF5Reader class, episode index
        schemas.py            # Pydantic response models
    requirements.txt
    pyproject.toml
```
