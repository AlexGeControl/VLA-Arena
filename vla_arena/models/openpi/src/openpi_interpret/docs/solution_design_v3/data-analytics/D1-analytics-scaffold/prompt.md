# Task D1: Analytics scaffold — module structure + data access

> Part of the [Data Analytics epic](../README.md). No dependencies; can proceed in parallel with D2a.

## Goal

Set up the `analytics/` Python package with:

1. Shared **constants** (token ranges, CMF pair specifications, silhouette group definitions).
2. Result **dataclasses** for per-timestep, per-episode, and global aggregation.
3. A thin **HDF5 reader** for analytics-specific data access (attention, Q-projections, cmf_attended, t-SNE).

## File placement

All files under **`openpi_interpret/analytics/`**:

| File | Purpose |
|------|---------|
| `__init__.py` | Package docstring |
| `constants.py` | `SAMPLED_LAYERS`, `TOKEN_RANGES`, `CMF_PAIRS`, `SILHOUETTE_GROUPS`, etc. |
| `types.py` | `TimestepCmf`, `TimestepSilhouette`, `EpisodeAnalytics`, `AnalyticsReport` |
| `reader.py` | `AnalyticsReader` (h5py wrapper), `scan_episodes()` directory scanner |
| `tests/__init__.py` | Test package |
| `tests/conftest.py` | Shared HDF5 test fixture with cmf_attended data |

## Constants

```python
SAMPLED_LAYERS = [0, 3, 6, 9, 12, 15, 17]
DEFAULT_CMF_LAYER = 17
NUM_HEADS = 8
HEAD_DIM = 256
NUM_ACTIONS = 50
NUM_SUFFIX = 51

TOKEN_RANGES = {
    "base_0_rgb": (0, 256), "left_wrist_0_rgb": (256, 512),
    "right_wrist_0_rgb": (512, 768), "language": (768, 816),
    "state": (816, 817), "action": (817, 867),
}

VISUAL_RANGE = (0, 768)
LANGUAGE_RANGE = (768, 816)

CMF_PAIRS = {
    "S_to_L": {"query_suffix_indices": [0], "target_key_range": (768, 816)},
    "S_to_V": {"query_suffix_indices": [0], "target_key_range": (0, 768)},
    "A_to_L": {"query_suffix_indices": list(range(1, 51)), "target_key_range": (768, 816)},
    "A_to_V": {"query_suffix_indices": list(range(1, 51)), "target_key_range": (0, 768)},
    "A_to_S": {"query_suffix_indices": list(range(1, 51)), "target_key_range": (816, 817)},
}

SILHOUETTE_GROUPS = {
    "visual": ["base_0_rgb", "left_wrist_0_rgb"],
    "language": ["language"],
    "action": ["state", "action"],
}
```

## AnalyticsReader

Thin h5py wrapper independent of the backend's `HDF5Reader`. Key methods:

- `get_attention(timestep, layer) -> ndarray [8, 51, 867]`
- `get_q_projections(timestep, layer) -> (q_prefix [816, 8, 256], q_suffix [51, 8, 256])`
- `get_cmf_attended(timestep, layer) -> (language [51, 8, 256], visual [51, 8, 256])`
- `get_tsne(timestep, layer) -> ndarray [867, 2]`
- `has_cmf_attended() -> bool` — graceful degradation for older HDF5 files
- `scan_episodes(data_dir) -> list[AnalyticsReader]` — module-level helper

Must resolve non-sequential timestep keys (same `_resolve_ts_key` pattern as backend).

## Result dataclasses

```python
@dataclass
class TimestepCmf:
    S_to_L: float; S_to_V: float; A_to_L: float; A_to_V: float; A_to_S: float

@dataclass
class TimestepSilhouette:
    score: float

@dataclass
class EpisodeAnalytics:
    episode_id: str; num_timesteps: int
    cmf_per_timestep: list[TimestepCmf]
    silhouette_per_timestep: list[TimestepSilhouette]
    # Properties: cmf_means, silhouette_mean

@dataclass
class AnalyticsReport:
    layer: int; episodes: list[EpisodeAnalytics]
    # Properties: global_cmf, global_silhouette
```

## Test fixture

`tests/conftest.py` generates a minimal HDF5 file with:
- 1 episode, 2 timesteps, 7 sampled layers
- Softmax-normalized attention `[8, 51, 867]`
- Q-projections: prefix `[816, 8, 256]`, suffix `[51, 8, 256]`
- cmf_attended: language `[51, 8, 256]`, visual `[51, 8, 256]`
- t-SNE coords `[867, 2]`

## Acceptance criteria

- [ ] `analytics/` importable as a Python package.
- [ ] Constants match the token map in [solution_design_v3 README](../../README.md).
- [ ] `AnalyticsReader` reads all HDF5 groups; resolves non-sequential timestep keys.
- [ ] `has_cmf_attended()` returns `False` for legacy files without the group.
- [ ] `scan_episodes()` returns empty list for non-existent directories (no crash).
- [ ] Test fixture generates valid HDF5; all reader methods return correct shapes.
