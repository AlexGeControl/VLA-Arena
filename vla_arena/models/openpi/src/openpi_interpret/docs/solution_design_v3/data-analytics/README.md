# Track D — Data Analytics Epic (OpenPI InterpreT)

This epic produces **offline interpretability metrics** from the HDF5 files created by Track B, without requiring GPU or the FastAPI backend.

## Purpose

Compute, from pre-extracted Pi-Zero model states:

- **Cross-Modal Fusion (CMF)** scores: attention-weighted cross-modal cosine similarity between query Q-projections and attended V-projections, per head in 256-d, aggregated across heads, queries, timesteps, and episodes.
- **Silhouette Coefficients**: cluster separation quality on t-SNE projections of the last layer's context-aware embeddings, using a 3-group modality clustering.

All computation runs **offline** in batch; the analytics module reads HDF5 and writes YAML reports.

## Architecture (end-to-end)

```
data/*.h5 (from Track B)
       │
       ├── attention/layer_XX           [8, 51, 867]    ──┐
       ├── q_projections/layer_XX/suffix [51, 8, 256]     ├── CMF scores
       ├── cmf_attended/layer_XX/*       [51, 8, 256]    ──┘
       │
       └── tsne/layer_XX                [867, 2]         ── Silhouette
       │
       ▼
analytics/run_analytics.py  ──►  analytics_report.yaml
```

**Data contracts consumed:**
- **Contract 1 (Track B → Track D)**: HDF5 files with attention, Q-projections, V-projection-based attended representations (`cmf_attended/`), and t-SNE coordinates.

**Data produced:**
- Structured YAML report with per-timestep, per-episode, and global metrics.

## Design Decisions (Validated)

| Decision | Choice | Rationale | Validated? |
|----------|--------|-----------|------------|
| CMF embedding space | Per-head 256-d | Cross-expert attention operates in shared head_dim=256, NOT concatenated 2048-d | Yes |
| CMF target embedding | V-projections (exact) | V has single KV head; faithfully represents what attention actually reads | Yes, Q-approx inflated cross-expert scores |
| CMF attention normalization | Intra-modality | Decouples "how much" from "how well aligned" | Yes |
| Silhouette clusters | 3 groups (visual, language, action+state) | Right wrist excluded (zero placeholder); state merged with action (both Expert 1) | Yes, improved from 0.33 to 0.37 |
| Silhouette input | t-SNE coords from layer 17 | Last layer captures final integration state | Yes |
| Compute location | Offline batch script | Keeps backend ML-free; metrics are deterministic | Yes |
| Output format | YAML | Human-readable, structured, easy to diff | Yes |

## Task Table

| Task | Focus | Key deliverables | Dependencies |
|------|-------|-----------------|--------------|
| **D1** | Analytics scaffold | Module structure, constants, types, HDF5 reader | None |
| **D2a** | V-projection capture + CMF attended | `capture.py` V intercept, `cmf_attended.py`, `serialize.py` extension | Track B (B2) |
| **D2b** | CMF score computation | `cmf.py`, per-head 256-d cosine, 5 pairs, aggregation | D1, D2a |
| **D3** | Silhouette coefficient | `silhouette.py`, 3-group labeling, sklearn integration | D1 |
| **D4** | CLI + YAML report | `run_analytics.py`, batch processing, structured output | D2b, D3 |

## Dependency Graph

```
D1  ──────────► D2b ──┐
D2a ──────────► D2b    ├──► D4
D1  ──► D3  ──────────┘
```

D1 and D2a can proceed in **parallel**. D2b depends on both D1 and D2a. D3 depends on D1. D4 integrates D2b and D3.

## CMF Score — Mathematical Definition

For each query token $q_k$ attending to target modality $M$ at head $h$:

1. Intra-modality normalized attention: $\hat{\alpha}_{k,j}^{(h)} = \alpha_{k,j}^{(h)} / \sum_{j' \in M} \alpha_{k,j'}^{(h)}$
2. Attended V-representation: $v_{\text{attended},k}^{(h)} = \sum_{j \in M} \hat{\alpha}_{k,j}^{(h)} \cdot V_j$ (V has single KV head, shared across groups)
3. Per-head CMF: $\text{CMF}^{(h)}(k) = \cos(Q_k^{(h)}, v_{\text{attended},k}^{(h)})$

Aggregation: mean over 8 heads → mean over query tokens → mean over timesteps → mean over episodes.

### The 5 CMF Pairs

| Pair | Query (suffix index) | Target modality (key range) | Expert crossing |
|------|---------------------|-----------------------------|-----------------|
| S → L | State (0) | Language (768–815) | Expert 1 → Expert 0 |
| S → V | State (0) | Visual (0–767) | Expert 1 → Expert 0 |
| A → L | Actions (1–50) | Language (768–815) | Expert 1 → Expert 0 |
| A → V | Actions (1–50) | Visual (0–767) | Expert 1 → Expert 0 |
| A → S | Actions (1–50) | State (816) | Expert 1 → Expert 1 |

## Silhouette Coefficient — Cluster Definition

| Cluster | Modalities | Token count | Rationale |
|---------|-----------|-------------|-----------|
| Visual | base_0_rgb + left_wrist_0_rgb | 512 | Right wrist excluded (zero placeholder for Pi-Zero on VLA-Arena) |
| Language | language | 48 | Global instruction tokens |
| Action | state + action | 51 | Both Expert 1, same Q/K/V projection weights |

## HDF5 Schema Extension (Contract 1 addition)

Track D requires a new `cmf_attended/` group added during extraction:

```
/timestep_NNN/cmf_attended/
  layer_{L:02d}/
    language      float32  (51, 8, 256)   # per suffix query, per head, head_dim
    visual        float32  (51, 8, 256)   # per suffix query, per head, head_dim
```

Pre-computed using V-projections (single KV head, `[867, 256]`) captured from the `BKGTS,BSKH->BTKGH` output einsum, with attention weights normalized within each target modality.

## Known Pitfall (Battle-Tested)

| # | Issue | Mitigation |
|---|-------|------------|
| **14** | Q-approximation inflates cross-expert CMF | Use exact V-projections for target embeddings. V has 1 KV head (shared across 8 groups); Q has 8 heads with separate per-expert weights, creating apparent cross-expert overlap that V does not exhibit. |
| **15** | V-projection capture counter alignment | The output einsum `BKGTS,BSKH->BTKGH` fires AFTER the logits einsum incremented the counter. Store V at `counter - 1` to align with Q's offset key. |
| **16** | Right wrist zero placeholder | Right wrist camera is all zeros for Pi-Zero on VLA-Arena. Exclude from silhouette clustering (256 noise tokens degrade cluster quality). |
| **17** | State singleton cluster | State is 1 token; a singleton cluster contributes silhouette = 0. Merge with action tokens (both Expert 1, same projection weights). |

## Environment

- **Python**: `openpi-vla-arena` conda env (Python 3.11)
- **Dependencies**: h5py, numpy, scikit-learn, PyYAML (all available in conda env)
- **GPU**: NOT required for analytics (CPU-only); required for extraction with V-capture (Track B extension)

## Code Layout

Under `openpi_interpret/analytics/`:

| Module | Role |
|--------|------|
| `__init__.py` | Package docstring |
| `constants.py` | Token ranges, CMF pair specs, sampled layers, silhouette groups |
| `types.py` | Result dataclasses: `TimestepCmf`, `TimestepSilhouette`, `EpisodeAnalytics`, `AnalyticsReport` |
| `reader.py` | Thin h5py wrapper for analytics reads (attention, Q-proj, cmf_attended, t-SNE) |
| `cmf.py` | CMF score computation: per-head cosine, 5 pairs, aggregation |
| `silhouette.py` | Silhouette Coefficient with 3-group clustering |
| `run_analytics.py` | CLI orchestrator + YAML report writer |

Extraction extensions under `openpi_interpret/extraction/`:

| Module | Change |
|--------|--------|
| `capture.py` | V-projection capture via `BKGTS,BSKH->BTKGH` intercept |
| `cmf_attended.py` | Intra-modality normalized attended representations from V-projections |
| `serialize.py` | `_write_cmf_attended()` for new HDF5 group |
| `extract_interpret_data.py` | Wiring: V-proj → cmf_attended computation → serialization |

## Task Prompts

1. [D1 — Analytics scaffold](D1-analytics-scaffold/prompt.md)
2. [D2a — V-projection capture + CMF attended representations](D2a-v-capture-cmf-attended/prompt.md)
3. [D2b — CMF score computation](D2b-cmf-computation/prompt.md)
4. [D3 — Silhouette coefficient](D3-silhouette/prompt.md)
5. [D4 — CLI orchestrator + report](D4-cli-report/prompt.md)
