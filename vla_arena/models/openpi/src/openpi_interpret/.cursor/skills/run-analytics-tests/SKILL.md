---
name: run-analytics-tests
description: >-
  Run the pytest test suite for the analytics pipeline. Validates CMF
  computation, silhouette scoring, and the full CLI report pipeline
  against synthetic HDF5 fixtures. Use when asked to run analytics tests,
  verify metrics, or check for regressions after code changes.
---

# Run Analytics Tests

## Run

```bash
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi_interpret

conda run -n openpi-vla-arena python -m pytest analytics/tests/ -v
```

## Expected Result

24 tests passing across 3 test files:
- `test_cmf.py` — cosine similarity, CMF from attended, direct A→S, all 5 pairs, extraction-time computation, V-shared-across-heads
- `test_silhouette.py` — label construction, 3-group clustering, right wrist exclusion, well-separated vs random coords
- `test_run_analytics.py` — episode processing, value ranges, full YAML pipeline round-trip

## What It Tests

- Cosine similarity correctness (identical, orthogonal, opposite vectors)
- CMF computation for all 5 cross-modal pairs
- Intra-modality attention normalization (zero-attention edge case)
- V-projection shared across heads (single KV head property)
- Silhouette 3-group label construction (611 included tokens, right wrist excluded)
- CLI orchestrator produces valid YAML with correct schema
- End-to-end: synthetic HDF5 → process_episode → YAML report → yaml.safe_load

## Run All Tests (analytics + backend)

```bash
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi_interpret

conda run -n openpi-vla-arena python -m pytest analytics/tests/ backend/tests/ -v
```

Expected: 57 tests (24 analytics + 33 backend), 0 failures.
