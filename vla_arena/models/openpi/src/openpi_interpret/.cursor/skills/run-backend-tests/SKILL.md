---
name: run-backend-tests
description: >-
  Run the pytest test suite for the OpenPI InterpreT backend. Validates
  all API endpoints against a test HDF5 fixture. Use when asked to run
  tests, verify the backend, or check for regressions.
---

# Run Backend Tests

## Run

```bash
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi_interpret/backend

conda run -n openpi-vla-arena python -m pytest tests/ -v
```

## Expected Result

33 tests passing across 3 test files:
- `test_episodes.py` — episode list, detail, camera image, token meta
- `test_attention.py` — attention row, breakdown, summary, parameter validation
- `test_embedding.py` — t-SNE points, neighbors, modality groups

## What It Tests

- All 9 REST API endpoints
- 404 for missing episodes and timesteps
- 422 for invalid layer/head/action parameters
- Attention row sums to ~1.0
- 867 token metadata entries
- 5 nearest neighbors per modality group
- Camera PNG encoding
