# Task C2: HDF5 Data Layer

> Part of the [Backend Development](../README.md) epic. Depends on [C1](../C1-fastapi-scaffold/prompt.md).

## Goal

A data access layer that indexes available episodes and provides efficient read access to HDF5 datasets (attention tensors, t-SNE coords, neighbors, camera images, metadata).

## Task

### Episode Index

On startup, scan `settings.data_dir` for `*.h5` files and build an in-memory index:

```python
# data/hdf5_reader.py

class EpisodeIndex:
    """In-memory index of available episodes, built on startup."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.episodes: dict[str, EpisodeSummary] = {}
        self._scan()

    def _scan(self):
        for h5_path in self.data_dir.glob("*.h5"):
            with h5py.File(h5_path, "r") as f:
                meta = f["/meta"]
                self.episodes[meta.attrs["episode_id"]] = EpisodeSummary(
                    episode_id=meta.attrs["episode_id"],
                    task_instruction=meta.attrs["task_instruction"],
                    num_timesteps=meta.attrs["num_timesteps"],
                    h5_path=h5_path,
                )
```

### HDF5 Reader

Provide typed accessor methods that open HDF5 files in read-only mode and return numpy arrays or Python objects:

```python
class HDF5Reader:
    """Read-only accessor for a single episode's HDF5 file."""

    def __init__(self, h5_path: Path):
        self.h5_path = h5_path

    def get_meta(self) -> EpisodeMeta: ...

    def get_camera_image(self, camera_name: str) -> np.ndarray: ...
        # Returns uint8 [H, W, 3]

    def get_token_meta(self, timestep: int) -> list[dict]: ...

    def get_attention(self, timestep: int, layer: int) -> np.ndarray: ...
        # Returns float32 [8, 51, 867]

    def get_tsne(self, timestep: int, layer: int) -> np.ndarray: ...
        # Returns float32 [867, 2]

    def get_neighbors(self, timestep: int, layer: int) -> np.ndarray: ...
        # Returns structured array [50, 5] with neighbor_index, distance

    def get_sampled_layers(self) -> list[int]: ...
```

Each method opens the HDF5 file, reads the requested dataset, and closes it. For production, consider keeping the file handle open for the request duration using a context manager.

### Pydantic Schemas (`data/schemas.py`)

```python
from pydantic import BaseModel

class EpisodeSummary(BaseModel):
    episode_id: str
    task_instruction: str
    num_timesteps: int

class EpisodeMeta(BaseModel):
    episode_id: str
    task_instruction: str
    num_timesteps: int
    instruction_tokens: list[str]
    sampled_layers: list[int]
    camera_names: list[str]

class TokenMeta(BaseModel):
    index: int
    type: str
    source: str
    patch_row: int | None = None
    patch_col: int | None = None
    token_text: str | None = None
    token_position: int | None = None

class AttentionBreakdownDetail(BaseModel):
    cameras: dict[str, list[float]]        # camera_name -> 256 weights
    camera_totals: dict[str, float]        # camera_name -> sum
    language_weights: list[float]
    language_total: float
    state_weight: float
    action_weights: list[float]
    action_total: float

class AttentionResponse(BaseModel):
    row: list[float]
    breakdown: AttentionBreakdownDetail

class AttentionSummary(BaseModel):
    modality_totals: dict[str, float]      # images, language, state, actions
    per_action: list[float]

class TsnePoint(BaseModel):
    index: int
    x: float
    y: float
    type: str
    source: str
    color: str

class TsneResponse(BaseModel):
    points: list[TsnePoint]

class SelectedPoint(BaseModel):
    index: int
    x: float
    y: float

class NearestNeighbor(BaseModel):
    index: int
    x: float
    y: float
    distance: float
    modality_group: str
    type: str
    source: str

class NeighborResponse(BaseModel):
    selected: SelectedPoint
    neighbors: list[NearestNeighbor]
```

### FastAPI Dependency

Register `EpisodeIndex` as a lifespan-scoped dependency:

```python
# main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.episode_index = EpisodeIndex(settings.data_dir)
    yield

def get_episode_index(request: Request) -> EpisodeIndex:
    return request.app.state.episode_index
```

## Acceptance Criteria

- [ ] `EpisodeIndex` scans and indexes all `.h5` files in `data_dir` on startup
- [ ] `HDF5Reader` can read attention, t-SNE, neighbors, camera, and metadata from a valid HDF5 file
- [ ] All Pydantic schemas validate correctly
- [ ] Missing episode or timestep returns appropriate 404
- [ ] No file handles are leaked (HDF5 files closed after each read)
