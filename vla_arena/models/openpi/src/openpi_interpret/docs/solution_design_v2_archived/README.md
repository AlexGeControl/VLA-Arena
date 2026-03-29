# OpenPI InterpreT — Solution Design v2

An interactive web application for visualizing the internal representations of the Pi-Zero VLA model, inspired by [VL-InterpreT](https://openaccess.thecvf.com/content/CVPR2022/html/Aflalo_VL-InterpreT_An_Interactive_Visualization_Tool_for_Interpreting_Vision-Language_Transformers_CVPR_2022_paper.html) (CVPR 2022, IntelLabs + MSRA). The tool helps researchers understand how the fine-tuned Pi-Zero model attends to visual observations, language instructions, and proprioceptive state when generating action trajectories.

> **Architecture reference**: [Pi-Zero Model Architecture](../../../../../docs/pi-zero/model_architecture.md)
> **Evaluation pipeline reference**: [Pi-Zero VLA-Arena Evaluation](../../../../../docs/pi-zero/metrics/README.md)

## Goals

1. **Action-centric attention exploration**: Visualize which image patches, language tokens, and state information each action token attends to across the dual-expert Gemma backbone.
2. **Semantic correspondence**: Visualize t-SNE projections of token embeddings (in the shared attention head space) to reveal cross-modal clustering and semantic proximity.
3. **Overfitting diagnosis**: Provide visual evidence for whether the fine-tuned model's attention patterns have collapsed.

## Three-Tier Architecture

```
┌─────────────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│  Data Extraction        │     │  FastAPI Backend      │     │  React Frontend       │
│  (Track B)              │────►│  (Track C)            │────►│  (Track A)            │
│                         │HDF5 │                       │REST │                       │
│  Pi-Zero inference      │     │  HDF5 reader          │JSON │  Attention heatmaps   │
│  + t-SNE pre-compute    │     │  Attention slicing    │     │  t-SNE scatter plot   │
│  + neighbor pre-compute │     │  Structured responses │     │  Shared controls      │
└─────────────────────────┘     └──────────────────────┘     └───────────────────────┘
```

Two data contracts:
- **Contract 1 (Extraction → Backend)**: HDF5 files — one per episode, with float32 attention tensors, pre-computed t-SNE, pre-computed nearest neighbors, Q-projections, camera images, and metadata.
- **Contract 2 (Backend → Frontend)**: REST API — JSON responses for episodes, attention breakdowns, t-SNE points, nearest neighbors; PNG responses for camera images.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Attention scope | **Suffix-only** (51 queries × 867 keys) | Focus on how conditional modalities influence action generation |
| Storage format | **HDF5** (float32, chunked) | Single file per episode, memory-mapped reads, hierarchical |
| Layer resolution | **Stride 3** — 7 layers: 0, 3, 6, 9, 12, 15, 17 | ~40% of full resolution, captures early/mid/late |
| t-SNE space | **Head-space Q-projection** (8 heads × 256 = 2048) | Architecturally principled shared space for both experts |
| t-SNE + neighbors | **Pre-computed during extraction** | Reproducible, zero ML deps on backend |
| Panel controls | **Shared globally** (layer, head, action) | Cross-panel brushing, no confusion |
| Backend | **FastAPI** (stateless, read-only) | Lightweight, auto-generated OpenAPI docs |

## Scope (First Attempt)

- **Model**: Pi-Zero (Pi0) only
- **Dataset**: `VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla` (602 episodes, 72K frames, 60 tasks)
- **Denoising step**: Single representative step (final, t~0.1)
- **Backbone focus**: Dual-expert Gemma backbone only

## Parallel Implementation Plan

Three independent development tracks sharing two data contracts.

### Epic: Data Extraction Pipeline (Track B)

Produces HDF5 files from Pi-Zero inference on L0 S episodes.

**[Full epic overview →](data-extraction/README.md)**

| Task | Deliverable | Dependency |
|------|------------|------------|
| [B1. Scaffold & Model Loading](data-extraction/B1-scaffold-model-loading/prompt.md) | CLI, model loading, inference verification | `uv sync` |
| [B2. Attention Capture](data-extraction/B2-attention-capture/prompt.md) | Monkey-patched hooks for attention + Q-projections | B1 |
| [B3. t-SNE & Neighbors](data-extraction/B3-tsne-neighbors/prompt.md) | Pre-computed t-SNE + nearest neighbors per action/modality | B2 |
| [B4. HDF5 Serialization](data-extraction/B4-hdf5-serialization/prompt.md) | HDF5 writer with all datasets | B2 + B3 |
| [B5. Validation](data-extraction/B5-validation/prompt.md) | End-to-end run, format checks, backend smoke test | B4 |

### Epic: Backend Development (Track C)

FastAPI service reading HDF5 files and serving structured REST responses.

**[Full epic overview →](backend-development/README.md)**

| Task | Deliverable | Dependency |
|------|------------|------------|
| [C1. FastAPI Scaffold](backend-development/C1-fastapi-scaffold/prompt.md) | Project setup, CORS, health check | None |
| [C2. HDF5 Data Layer](backend-development/C2-hdf5-data-layer/prompt.md) | Episode index, HDF5 reader, Pydantic schemas | C1 |
| [C3. Episode & Metadata API](backend-development/C3-episode-metadata-api/prompt.md) | `/episodes`, `/camera`, `/token-meta` | C2 |
| [C4. Attention API](backend-development/C4-attention-api/prompt.md) | `/attention` (row + breakdown), `/attention/summary` | C2 |
| [C5. Embedding API](backend-development/C5-embedding-api/prompt.md) | `/tsne` (points), `/tsne/neighbors` (pre-computed reads) | C2 |

### Epic: Frontend Development (Track A)

React TypeScript SPA consuming the REST API.

**[Full epic overview →](frontend-development/README.md)**

| Task | Deliverable | Dependency |
|------|------------|------------|
| [A1. API Client & Scaffold](frontend-development/A1-api-client-scaffold/prompt.md) | TypeScript interfaces, API client, mock setup | API contract |
| [A2. Navigation Shell](frontend-development/A2-navigation-shell/prompt.md) | Home, episode page, shared controls, slider | Task A1 |
| [A3. Attention View](frontend-development/A3-attention-view/prompt.md) | Language heatmap, image overlay, summary bar | Task A2 |
| [A4. Embedding View](frontend-development/A4-embedding-view/prompt.md) | t-SNE scatter, nearest neighbors, tooltips | Task A2 |
| [A5. Polish & Integration](frontend-development/A5-polish/prompt.md) | Cross-panel brushing, pop-ups, responsive layout | Tasks A3 & A4 |

### Sequencing

```
Track B (Extraction):  B1 ► B2 ► B3 ► B4 ► B5
                                            │ (HDF5 files)
                                            ▼
Track C (Backend):     C1 ► C2 ──► C3, C4, C5
                            │                │ (REST API)
                   (API contract)            ▼
Track A (Frontend):    Task A1 ► Task A2 ► Task A3 ──┐
                                        └► Task A4 ──┤► Task A5
```

- **Track B** and **Track C** share Contract 1 (HDF5 format)
- **Track C** and **Track A** share Contract 2 (REST API)
- Track A can start with mock API (MSW) before Track C delivers
- Track C can start with test HDF5 files before Track B delivers

## Token Map

| Token Range | Count | Modality | Expert | Color |
|-------------|-------|----------|--------|-------|
| 0–255 | 256 | Base camera patches | 0 | Blue |
| 256–511 | 256 | Left wrist patches | 0 | Cyan |
| 512–767 | 256 | Right wrist patches | 0 | Teal |
| 768–815 | ≤48 | Language tokens | 0 | Orange |
| 816 | 1 | Proprioceptive state | 1 | Green |
| 817–866 | 50 | Action tokens | 1 | Red |

## File Placement

```
openpi_interpret/
  extraction/           # Track B: Python extraction scripts
  backend/              # Track C: FastAPI application
  frontend/             # Track A: React TypeScript SPA
  data/                 # HDF5 episode files (gitignored)
```

## Backlog

- **Attention Head Summary Grid**: Layer×head heatmap of per-head metrics
- **Denoising Step Slider**: Compare attention at t=1.0 vs t=0.1
- **Full Prefix Attention**: V2V/V2L/L2V/L2L for prefix-internal analysis
- **Re-computable t-SNE**: Backend endpoint to re-run t-SNE from stored Q-projections with custom perplexity
