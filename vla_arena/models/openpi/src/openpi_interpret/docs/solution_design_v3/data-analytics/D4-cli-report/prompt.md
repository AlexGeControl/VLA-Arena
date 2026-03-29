# Task D4: CLI orchestrator + YAML report

> Part of the [Data Analytics epic](../README.md). Depends on [D2b](../D2b-cmf-computation/prompt.md) and [D3](../D3-silhouette/prompt.md).

## Goal

Implement **`analytics/run_analytics.py`** — a CLI entry point that batch-processes all episode HDF5 files, computes CMF and Silhouette metrics, and writes a structured YAML report.

## CLI interface

```bash
conda run -n openpi-vla-arena python \
  src/openpi_interpret/analytics/run_analytics.py \
    --data-dir src/openpi_interpret/data \
    --output analytics_report.yaml \
    --layer 17
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `src/openpi_interpret/data` | Directory containing .h5 episode files |
| `--output` | `analytics_report.yaml` | Output report path |
| `--layer` | `17` | Transformer layer (must be in sampled set) |
| `--log-level` | `INFO` | Logging verbosity |

## Processing pipeline

1. Validate `--layer` is in `[0, 3, 6, 9, 12, 15, 17]`.
2. `scan_episodes(data_dir)` → list of `AnalyticsReader`.
3. For each episode:
   - Check `has_cmf_attended()` — if False, warn and skip CMF (silhouette still computed).
   - For each timestep:
     - Load `q_suffix` and `cmf_attended` → `compute_all_cmf()` → `TimestepCmf`
     - Load `tsne` → `compute_silhouette()` → `TimestepSilhouette`
   - Aggregate into `EpisodeAnalytics`.
4. Build `AnalyticsReport`.
5. Serialize to YAML with `report_to_dict()` → `write_yaml()`.
6. Log summary.

## YAML report schema

```yaml
metadata:
  generated_at: "2026-03-29T22:22:26+00:00"
  layer: 17
  num_episodes: 5

global:
  silhouette: 0.3700
  cmf:
    S_to_L: -0.0221
    S_to_V: -0.0329
    A_to_L: 0.0024
    A_to_V: -0.0181
    A_to_S: 0.2487

episodes:
  - episode_id: ep_000000
    num_timesteps: 13
    cmf:
      S_to_L: { mean: -0.018, per_timestep: [...] }
      S_to_V: { mean: -0.032, per_timestep: [...] }
      A_to_L: { mean: 0.013, per_timestep: [...] }
      A_to_V: { mean: -0.016, per_timestep: [...] }
      A_to_S: { mean: 0.244, per_timestep: [...] }
    silhouette: { mean: 0.357, per_timestep: [...] }
```

All float values rounded to 4 decimal places.

## Graceful degradation

If an HDF5 file lacks `cmf_attended/`:
- Log a warning with the episode ID.
- Skip CMF computation for that episode.
- Still compute silhouette.
- Omit `cmf` from the episode's YAML entry and from `global.cmf`.

## File placement

**`openpi_interpret/analytics/run_analytics.py`**

Key functions:
- `parse_args() -> Namespace`
- `process_episode(reader, layer) -> EpisodeAnalytics`
- `build_report(episodes, layer) -> AnalyticsReport`
- `report_to_dict(report) -> dict`
- `write_yaml(report_dict, output_path)`
- `main()`

## Test cases (`tests/test_run_analytics.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_episode_produces_metrics` | process test fixture | 2 timesteps of CMF + silhouette |
| `test_cmf_values_in_range` | all CMF scores | in [−1, 1] |
| `test_silhouette_values_in_range` | all silhouette scores | in [−1, 1] |
| `test_full_pipeline_writes_yaml` | end-to-end | valid YAML with correct structure |

## Performance

- 5 episodes × ~12 timesteps = ~60 timesteps.
- CMF: matrix multiplications on `[51, 8, 256]` tensors — negligible.
- Silhouette: pairwise distances on 611 × 2 coords — ~1ms per call.
- Total runtime: **< 5 seconds** for 5 episodes.

## Acceptance criteria

- [ ] CLI validates `--layer` against sampled set; exits 1 on invalid.
- [ ] Empty `--data-dir` produces clear error message, exits 1.
- [ ] YAML output matches schema: `metadata`, `global`, `episodes` keys present.
- [ ] `global.cmf` contains all 5 pairs; `global.silhouette` is a scalar.
- [ ] Per-episode `cmf` has `mean` + `per_timestep` list for each pair.
- [ ] Graceful degradation: HDF5 without `cmf_attended` → warning, CMF skipped, silhouette computed.
- [ ] Summary logged to stdout with global scores.
- [ ] All tests pass; YAML round-trips through `yaml.safe_load()`.
