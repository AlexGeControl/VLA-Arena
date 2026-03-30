---
name: run-analytics
description: >-
  Run the offline analytics pipeline on extracted HDF5 data. Computes
  Cross-Modal Fusion (CMF) scores and Silhouette Coefficients, outputs
  a structured YAML report. Use when asked to compute metrics, generate
  an analytics report, or re-run analytics after re-extraction.
---

# Run Analytics

## Prerequisites

- conda env `openpi-vla-arena` with scikit-learn and PyYAML installed
- Extracted HDF5 files in `src/openpi_interpret/data/` (run `extract-model-states` skill first)
- For CMF scores: HDF5 files must contain `cmf_attended/` groups (requires extraction with V-projection capture)

## Run Analytics

```bash
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi_interpret

conda run -n openpi-vla-arena python analytics/run_analytics.py \
  --data-dir data \
  --output data/analytics_report.yaml \
  --layer 17
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `src/openpi_interpret/data` | Directory containing .h5 episode files |
| `--output` | `analytics_report.yaml` | Path for the YAML report output |
| `--layer` | `17` | Transformer layer for metrics (must be in `[0,3,6,9,12,15,17]`) |
| `--log-level` | `INFO` | Logging verbosity |

## Expected Output

- YAML report with global, per-episode, and per-timestep metrics
- 5 CMF scores: S→L, S→V, A→L, A→V, A→S
- 1 Silhouette Coefficient (3-group: visual, language, action+state)
- Runtime: < 5 seconds for 5 episodes

## Graceful Degradation

If HDF5 files lack `cmf_attended/` (older extraction without V-projection capture):
- Silhouette scores are still computed
- CMF scores are skipped with a warning
- Re-run extraction with the `extract-model-states` skill to get V-projections

## Backfill CMF Attended (without re-extraction)

If HDF5 files have `q_projections/` but no `cmf_attended/`, you can backfill
using Q-projection approximation (less accurate than V-projections):

```bash
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi_interpret

conda run -n openpi-vla-arena python -c "
import h5py, numpy as np, json
from pathlib import Path
from extraction.cmf_attended import compute_cmf_attended

for p in sorted(Path('data').glob('*.h5')):
    with h5py.File(p, 'a') as f:
        layers = json.loads(f['meta'].attrs['sampled_layers'])
        for ts_key in sorted(k for k in f if k.startswith('timestep_')):
            if f'{ts_key}/cmf_attended' in f:
                continue
            cmf_grp = f[ts_key].create_group('cmf_attended')
            for layer in layers:
                lk = f'layer_{layer:02d}'
                attn = np.array(f[f'{ts_key}/attention/{lk}'], dtype=np.float32)
                q_prefix = np.array(f[f'{ts_key}/q_projections/{lk}/prefix'], dtype=np.float32)
                # Q-approx: use q_prefix as proxy for V (shape [816, 8, 256])
                # For exact V, re-run extraction instead
                result = compute_cmf_attended(attn, q_prefix[:, 0, :])  # take head 0 as [816, 256]
                lg = cmf_grp.create_group(lk)
                lg.create_dataset('language', data=result['language'], compression='gzip')
                lg.create_dataset('visual', data=result['visual'], compression='gzip')
    print(f'Backfilled {p.name}')
"
```

Note: this uses a single Q-head as proxy. For accurate V-based CMF, re-extract with the current pipeline.
