# Backend API (Track C)

FastAPI service that reads pre-computed Pi-Zero model states from HDF5 files
and serves them as structured JSON via REST API. Zero ML dependencies — pure
reads + lightweight numpy slicing.

## Architecture

```
HDF5 files (from extraction) → HDF5Reader (Repository) → Routers → JSON
```

- `app/data/hdf5_reader.py` — Repository pattern. All HDF5 access encapsulated here.
- `app/data/schemas.py` — Pydantic v2 response models.
- `app/data/constants.py` — Token ranges, sampled layers, modality colors.
- `app/routers/episodes.py` — Episode listing, metadata, camera images, token metadata.
- `app/routers/attention.py` — Attention row + modality breakdown, attention summary.
- `app/routers/embedding.py` — t-SNE points with colors, nearest neighbors.

## API Endpoints

| Endpoint | Response |
|----------|----------|
| `GET /api/health` | `{"status": "ok"}` |
| `GET /api/episodes` | Episode list |
| `GET /api/episodes/{id}` | Full metadata |
| `GET /api/episodes/{id}/camera/{name}?timestep=N` | PNG image |
| `GET /api/episodes/{id}/timesteps/{t}/token-meta` | 867 token entries |
| `GET /api/episodes/{id}/timesteps/{t}/attention?layer=L&head=H&action=A` | Attention row + breakdown |
| `GET /api/episodes/{id}/timesteps/{t}/attention/summary?layer=L&head=H` | Modality totals |
| `GET /api/episodes/{id}/timesteps/{t}/tsne?layer=L` | 867 t-SNE points |
| `GET /api/episodes/{id}/timesteps/{t}/tsne/neighbors?layer=L&action=A` | 5 nearest neighbors |

## Usage

```bash
INTERPRET_DATA_DIR=../data conda run -n openpi-vla-arena \
  uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Tests

```bash
conda run -n openpi-vla-arena python -m pytest tests/ -v
```
