# OpenPI InterpreT — Solution Design v3 (Battle-Tested)

An interactive web service for visualizing Pi-Zero's internal representations, inspired by [VL-InterpreT](https://openaccess.thecvf.com/content/CVPR2022/html/Aflalo_VL-InterpreT_An_Interactive_Visualization_Tool_for_Interpreting_Vision-Language_Transformers_CVPR_2022_paper.html) (CVPR 2022, IntelLabs + MSRA).

This is the **refined, reproducible** plan incorporating all lessons learned from the end-to-end implementation. The v2 plan (predecessor) is preserved for historical reference.

> **Architecture reference**: [Pi-Zero Model Architecture](docs/reference/pi-zero/model_architecture.md)
> **Evaluation pipeline reference**: [Pi-Zero VLA-Arena Evaluation](../../../../../docs/pi-zero/metrics/README.md)

## Goals

1. **Action-centric attention exploration**: Visualize cross-attention from each action token to all conditional modalities (image patches, language tokens, state) across the dual-expert Gemma backbone — per layer, per head.
2. **Semantic correspondence**: Visualize t-SNE projections of token embeddings in the shared attention head space — per layer only (head has no effect on t-SNE). Show nearest neighbors per modality for the selected action token.
3. **Overfitting diagnosis**: Provide visual evidence for whether attention patterns have collapsed.

## Three-Tier Architecture

```
extraction/  (Track B)    -->  data/*.h5      -->  backend/  (Track C)  -->  frontend/  (Track A)
Python + JAX + GPU             HDF5 files         FastAPI + h5py          React + TypeScript
```

Two data contracts:
- **Contract 1 (Extraction --> Backend)**: HDF5 files with per-timestep attention, t-SNE, neighbors, camera images, and metadata.
- **Contract 2 (Backend --> Frontend)**: REST API with JSON responses and PNG camera images.

## Design Decisions (Validated)

| Decision | Choice | Rationale | Validated? |
|----------|--------|-----------|------------|
| Attention scope | Suffix-only (51 queries x 867 keys) | Focus on action-to-condition cross-attention | Yes |
| Storage format | HDF5 (float32, per-episode files) | Single file, hierarchical, memory-mapped | Yes |
| Layer resolution | Stride 3 -- 7 layers: 0, 3, 6, 9, 12, 15, 17 | ~40% of full resolution, early/mid/late coverage | Yes |
| Timestep resolution | **Stride 10** (1.0s at 10Hz) | ~11-13 timesteps/episode, good balance | Revised from stride 40 |
| t-SNE space | Head-space Q-projection (8 heads x 256 = 2048) | Architecturally principled shared space | Yes, produced meaningful clusters |
| t-SNE + neighbors | Pre-computed during extraction | Reproducible, zero ML deps on backend | Yes |
| Camera images | **Per-timestep** | Robot moves between frames | Revised from per-episode |
| Panel controls | Shared globally (layer, head, action) | Cross-panel brushing, no confusion | Yes |
| Backend | FastAPI, stateless, read-only, `allow_origins=["*"]` | Lightweight, LAN-accessible | Revised CORS from localhost-only |
| Checkpoint loading | **Direct cache path** with `local_files_only=True` | Avoids corrupted HF snapshots | Revised from `snapshot_download` |

## Scope

- **Model**: Pi-Zero (Pi0) fine-tuned on VLA-Arena
- **Dataset**: `VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla` (602 episodes, 72K frames, 60 tasks)
- **Extraction scope**: 3 episodes, stride 10 (~37 timesteps total)
- **Denoising step**: Final step (t=0.1)
- **Backbone focus**: Dual-expert Gemma backbone (18 layers, 8 heads)

## HDF5 Schema (Contract 1)

```
{episode_id}.h5
  /meta
    @episode_id          (str)
    @task_instruction    (str)
    @num_timesteps       (int)
    @instruction_tokens  (JSON str)
    @sampled_layers      (JSON str: [0,3,6,9,12,15,17])
  /timestep_NNN/                       # NNN = raw frame index (not sequential)
    cameras/
      base_0_rgb         (224,224,3)   uint8   # per-timestep, NOT per-episode
      left_wrist_0_rgb   (224,224,3)   uint8
      right_wrist_0_rgb  (224,224,3)   uint8   # masked (zeros) if unavailable
    token_meta           (JSON str)             # 867 entries, type="image_patch" (not "image")
    attention/
      layer_XX           (8,51,867)    float32  # per-head cross-attention
    tsne/
      layer_XX           (867,2)       float32  # per-layer only, NOT per-head
    neighbors/
      layer_XX           (50,5)        compound # per-layer only, NOT per-head
                                                # fields: neighbor_index(i4), distance(f4)
    q_projections/                              # backlog asset
      layer_XX/
        prefix           (816,8,256)   float32
        suffix           (51,8,256)    float32
```

**Key points**:
- Timestep group names use raw frame indices (e.g. `timestep_000`, `timestep_010`, `timestep_020`), NOT sequential indices. The backend maps sequential index 0,1,2... to actual group names.
- Camera images are per-timestep (the robot moves).
- Token type for image patches is `"image_patch"` (not `"image"`).
- Attention is per-layer per-head. t-SNE and neighbors are per-layer only.

## REST API (Contract 2)

| Endpoint | Response | Notes |
|----------|----------|-------|
| `GET /api/health` | `{"status":"ok"}` | |
| `GET /api/episodes` | `[{episode_id, task_instruction, num_timesteps}]` | |
| `GET /api/episodes/{id}` | `{..., instruction_tokens, sampled_layers, camera_names}` | |
| `GET /api/episodes/{id}/camera/{cam}?timestep=N` | `image/png` | **timestep required** for correct frame |
| `GET /api/episodes/{id}/timesteps/{t}/token-meta` | `[{index, type, source, ...}]` | t = sequential index |
| `GET /api/episodes/{id}/timesteps/{t}/attention?layer=L&head=H&action=A` | `{row, breakdown}` | Per-head |
| `GET /api/episodes/{id}/timesteps/{t}/attention/summary?layer=L&head=H` | `{modality_totals, per_action}` | Per-head |
| `GET /api/episodes/{id}/timesteps/{t}/tsne?layer=L` | `{points}` | Per-layer only |
| `GET /api/episodes/{id}/timesteps/{t}/tsne/neighbors?layer=L&action=A` | `{selected, neighbors}` | Per-layer only |

## Lessons Learned (12 Issues)

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | HF checkpoint corruption | Aborted `snapshot_download` left truncated Zstd blob | Use direct cache path; `local_files_only=True` |
| 2 | `while_loop` tracing | `sample_actions` JIT-traces loop body; patches don't fire | Separate capture pass with `jax.disable_jit()` |
| 3 | `nn.remat` tracing | `Block` wrapped in remat; dict stores capture tracers | `jax.debug.callback` materializes arrays from traced code |
| 4 | SigLIP softmax interference | SigLIP ViT also calls `jax.nn.softmax` (4D tensors) | Only capture 5D tensors (Gemma backbone format) |
| 5 | Layer counter offset | Extra einsum/softmax calls before first Gemma layer | Auto-detect offset from captured key ranges |
| 6 | GPU OOM | `disable_jit()` mode accumulates memory over timesteps | `gc.collect()` + `jax.clear_caches()` per timestep |
| 7 | Token type naming | Extraction wrote `"image"` instead of `"image_patch"` | Fixed in `build_token_meta()` |
| 8 | Timestep key mismatch | HDF5 uses frame indices; frontend sends sequential | Backend `_resolve_ts_key()` maps sequential to actual |
| 9 | Camera images per-episode | Robot moves; single image is wrong for other timesteps | Cameras stored per-timestep |
| 10 | Camera detection | `_detect_cameras` checked absent `/cameras/` group | Fallback to first timestep's cameras |
| 11 | ImagePatchPopup timestep | Popup showed first frame instead of current | Pass `timestep` prop through component chain |
| 12 | Policy attribute name | `policy.model` doesn't exist | Correct attribute is `policy._model` |
| 13 | t-SNE dark token contrast | CMU palette uses dark tones (navy, teal); invisible on dark/transparent SVG background | White SVG background, black highlight strokes, dark gray neighbor lines |

## Implementation Plan

### Proven Sprint Sequencing

```
Sprint 1: Track C (Backend C1-C5) + Track A scaffold (A1-A2)
    - No GPU needed, parallel agents
    - Backend creates test HDF5 fixture for development
    - Frontend uses MSW mocks or test fixture via backend

Sprint 2: Track B (Data Extraction B1-B5)
    - GPU required (7GB VRAM)
    - Produces real HDF5 data files
    - Backend smoke test validates output

Sprint 3: Track A visualization (A3-A4-A5)
    - Connect to live backend with real data
    - A3 and A4 in parallel (separate agents)
    - A5 after both complete

Integration: Human-in-the-loop testing
    - Timestamp slider, camera updates, cross-panel brushing
    - Fix issues 8-11 discovered during manual testing
```

### Epic: Data Extraction (Track B)

**[Full epic overview -->](data-extraction/README.md)**

| Task | Deliverable | Key Pitfalls |
|------|------------|-------------|
| [B1. Scaffold & Model Loading](data-extraction/B1-scaffold-model-loading/prompt.md) | CLI, model loading | #1 (checkpoint path), #12 (`_model` attribute) |
| [B2. Attention Capture](data-extraction/B2-attention-capture/prompt.md) | Monkey-patched hooks | #2 (while_loop), #3 (remat), #4 (SigLIP), #5 (offset) |
| [B3. t-SNE & Neighbors](data-extraction/B3-tsne-neighbors/prompt.md) | Pre-computed embeddings | |
| [B4. HDF5 Serialization](data-extraction/B4-hdf5-serialization/prompt.md) | HDF5 writer | #7 (token type), #9 (per-timestep cameras) |
| [B5. Validation](data-extraction/B5-validation/prompt.md) | End-to-end run | #6 (OOM fix) |

### Epic: Backend Development (Track C)

**[Full epic overview -->](backend-development/README.md)**

| Task | Deliverable | Key Pitfalls |
|------|------------|-------------|
| [C1. FastAPI Scaffold](backend-development/C1-fastapi-scaffold/prompt.md) | Project setup, CORS | CORS `allow_origins=["*"]` for LAN |
| [C2. HDF5 Data Layer](backend-development/C2-hdf5-data-layer/prompt.md) | Reader, schemas | #8 (`_resolve_ts_key`), #10 (camera detection) |
| [C3. Episode & Metadata API](backend-development/C3-episode-metadata-api/prompt.md) | Episodes, camera, token-meta | Camera endpoint needs `?timestep=N` |
| [C4. Attention API](backend-development/C4-attention-api/prompt.md) | Attention row + breakdown | |
| [C5. Embedding API](backend-development/C5-embedding-api/prompt.md) | t-SNE points, neighbors | |

### Epic: Frontend Development (Track A)

**[Full epic overview -->](frontend-development/README.md)**

| Task | Deliverable | Key Pitfalls |
|------|------------|-------------|
| [A1. API Client & Scaffold](frontend-development/A1-api-client-scaffold/prompt.md) | TypeScript interfaces, API client | `getCameraImageUrl` needs timestep param |
| [A2. Navigation Shell](frontend-development/A2-navigation-shell/prompt.md) | Home, episode page, shared controls | |
| [A3. Attention View](frontend-development/A3-attention-view/prompt.md) | Language heatmap, image overlay | Pass `timestep` to `ImageAttention` |
| [A4. Embedding View](frontend-development/A4-embedding-view/prompt.md) | t-SNE scatter, neighbors | |
| [A5. Polish & Integration](frontend-development/A5-polish/prompt.md) | Cross-panel brushing, pop-ups | #11 (ImagePatchPopup timestep) |

## Visual Theme — CMU Brand Alignment

The visual design follows [CMU Brand Standards](https://brand.cmu.edu/visual-identity/colors), using Core Colors for the UI chrome and a curated subset of Secondary Colors for data visualization.

### UI Chrome (Core Colors)

| Element | Color | CMU Name | Hex |
|---------|-------|----------|-----|
| Header bars | Carnegie Red | Core | `#C41230` |
| Page background | White | Core | `#FFFFFF` |
| Card/panel borders | Steel Gray | Core | `#E0E0E0` |
| Text primary | Black | Core | `#000000` |
| Text secondary | Iron Gray | Core | `#6D6E71` |
| Loading spinners | Carnegie Red | Core | `#C41230` |
| Attention heatmap ramp | Carnegie Red | Core | `rgba(196,18,48,alpha)` |

### Data Visualization (Token Modality Colors)

| Modality | Color | CMU Name | Hex | Rationale |
|----------|-------|----------|-----|-----------|
| Base camera | Blue Thread | Tartan | `#043673` | Cool tone for spatial data |
| Left wrist camera | Highlands Sky Blue | Tartan | `#007BC0` | Lighter blue, distinct from base |
| Right wrist camera | Hornbostel Teal | Campus | `#1F4C4C` | Dark teal, third camera distinction |
| Language tokens | Gold Thread | Tartan | `#FDB515` | Warm, high contrast on dark backgrounds |
| Proprioceptive state | Green Thread | Tartan | `#009647` | Naturally "grounded" for physical state |
| Action tokens | Carnegie Red | Core | `#C41230` | Brand hero color for the analysis focus |

### t-SNE Scatter Plot Style (Pitfall #13)

The CMU token colors are predominantly dark tones. To ensure visibility:
- **Background**: White (`#FFFFFF`) with Steel Gray border
- **Point strokes**: Black (`#000000`) for selected/highlighted tokens (not white)
- **Neighbor rings**: Black strokes with 2px width
- **Neighbor lines**: Iron Gray at 60% opacity (`rgba(109,110,113,0.6)`)
- **Default point opacity**: 0.6 (lets the white background show through for depth)

## Token Map

| Token Range | Count | Type | Expert | Color |
|-------------|-------|------|--------|-------|
| 0-255 | 256 | `image_patch` | 0 | Blue Thread `#043673` |
| 256-511 | 256 | `image_patch` | 0 | Sky Blue `#007BC0` |
| 512-767 | 256 | `image_patch` | 0 | Hornbostel Teal `#1F4C4C` |
| 768-815 | <=48 | `language` | 0 | Gold Thread `#FDB515` |
| 816 | 1 | `state` | 1 | Green Thread `#009647` |
| 817-866 | 50 | `action` | 1 | Carnegie Red `#C41230` |

SigLIP ViT-So400m/14 patchification: 224x224 image --> Conv(kernel=14, stride=14) --> 16x16 grid --> `jnp.reshape` row-major --> token index = row*16 + col.

## Environment Requirements

| Dependency | Version | Track |
|-----------|---------|-------|
| Python | 3.11 | B, C |
| JAX | 0.5.3 | B only |
| h5py | 3.13+ | B, C |
| scikit-learn | 1.8+ | B only |
| FastAPI | 0.115+ | C only |
| uvicorn | 0.30+ | C only |
| Node.js | 20 LTS | A only |
| React | 18 | A only |
| TypeScript | strict mode | A only |
| GPU | 7GB+ VRAM | B only |

## File Placement

```
openpi_interpret/
  extraction/           # Track B (6 files, 1254 lines)
  backend/              # Track C (16 files, 1311 lines)
  frontend/             # Track A (29 files, 1553 lines)
  data/                 # HDF5 episode files (~1.1 GB for 3 episodes)
  docs/
    solution_design_v3/ # This plan
    reference/          # Pi-Zero architecture docs + diagrams
  .cursor/rules/        # 8 rule files (4 standards + 4 agent roles)
```
