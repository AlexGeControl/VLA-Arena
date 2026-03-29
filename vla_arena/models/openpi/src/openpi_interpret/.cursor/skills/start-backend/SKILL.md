---
name: start-backend
description: >-
  Start the FastAPI backend server for OpenPI InterpreT. Serves attention
  weights, t-SNE embeddings, and camera images from HDF5 files via REST API.
  Use when asked to start the backend, API server, or visualization service.
---

# Start Backend

## Prerequisites

- conda env `openpi-vla-arena` with fastapi, uvicorn, h5py installed
- HDF5 data files in `src/openpi_interpret/data/` (from extraction)

## Start

```bash
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi_interpret/backend

INTERPRET_DATA_DIR=../data conda run -n openpi-vla-arena \
  uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Verify

```bash
curl http://localhost:8080/api/health
# Should return: {"status":"ok"}

curl http://localhost:8080/api/episodes
# Should return list of episodes
```

## Stop

```bash
fuser -k 8080/tcp
```

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/api/health` | Liveness check |
| `/api/episodes` | List episodes |
| `/api/episodes/{id}` | Episode metadata |
| `/api/episodes/{id}/camera/{cam}?timestep=N` | Camera PNG |
| `/api/episodes/{id}/timesteps/{t}/attention?layer=L&head=H&action=A` | Attention |
| `/api/episodes/{id}/timesteps/{t}/tsne?layer=L` | t-SNE points |
| `/api/episodes/{id}/timesteps/{t}/tsne/neighbors?layer=L&action=A` | Neighbors |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INTERPRET_DATA_DIR` | `../data` | Path to HDF5 episode files |
| `INTERPRET_HOST` | `0.0.0.0` | Bind address |
| `INTERPRET_PORT` | `8080` | Port |
