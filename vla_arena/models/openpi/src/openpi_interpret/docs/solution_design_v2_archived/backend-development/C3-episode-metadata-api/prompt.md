# Task C3: Episode & Metadata API

> Part of the [Backend Development](../README.md) epic. Depends on [C2](../C2-hdf5-data-layer/prompt.md).

## Goal

REST endpoints for browsing episodes, fetching metadata, serving camera images, and retrieving token metadata.

## Task

### Endpoints

Create `app/routers/episodes.py`:

**`GET /api/episodes`** — List all available episodes.

```python
@router.get("/api/episodes", response_model=list[EpisodeSummary])
async def list_episodes(index: EpisodeIndex = Depends(get_episode_index)):
    return sorted(index.episodes.values(), key=lambda e: e.episode_id)
```

Response: `[{ "episode_id": "ep_042", "task_instruction": "pick up...", "num_timesteps": 4 }]`

**`GET /api/episodes/{episode_id}`** — Episode detail with instruction tokens and configuration.

```python
@router.get("/api/episodes/{episode_id}", response_model=EpisodeMeta)
async def get_episode(episode_id: str, index: EpisodeIndex = Depends(get_episode_index)):
    entry = index.episodes.get(episode_id)
    if not entry:
        raise HTTPException(404, f"Episode {episode_id} not found")
    reader = HDF5Reader(entry.h5_path)
    return reader.get_meta()
```

**`GET /api/episodes/{episode_id}/camera/{camera_name}`** — Camera image as PNG.

```python
from fastapi.responses import Response
from PIL import Image
from io import BytesIO

@router.get("/api/episodes/{episode_id}/camera/{camera_name}")
async def get_camera_image(episode_id: str, camera_name: str, ...):
    img_array = reader.get_camera_image(camera_name)  # uint8 [H, W, 3]
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
```

**`GET /api/episodes/{episode_id}/timesteps/{timestep}/token-meta`** — Token metadata for a timestep.

```python
@router.get("/api/episodes/{episode_id}/timesteps/{timestep}/token-meta",
            response_model=list[TokenMeta])
async def get_token_meta(episode_id: str, timestep: int, ...):
    return reader.get_token_meta(timestep)
```

### Validation

- `episode_id` must exist in the index (404 otherwise)
- `camera_name` must be one of `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` (422 otherwise)
- `timestep` must be in range `[0, num_timesteps)` (404 otherwise)

## Acceptance Criteria

- [ ] `GET /api/episodes` returns the full episode list
- [ ] `GET /api/episodes/{id}` returns metadata with instruction_tokens and sampled_layers
- [ ] `GET /api/episodes/{id}/camera/base_0_rgb` returns a valid PNG image
- [ ] `GET /api/episodes/{id}/camera/right_wrist_0_rgb` returns 404 if camera is masked
- [ ] `GET /api/episodes/{id}/timesteps/0/token-meta` returns 867 token entries
- [ ] Invalid episode_id returns 404
- [ ] Invalid timestep returns 404
