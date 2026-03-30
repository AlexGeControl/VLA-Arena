# OpenPI InterpreT

Interactive web service for visualizing Pi-Zero's internal representations,
inspired by [VL-InterpreT](https://openaccess.thecvf.com/content/CVPR2022/html/Aflalo_VL-InterpreT_An_Interactive_Visualization_Tool_for_Interpreting_Vision-Language_Transformers_CVPR_2022_paper.html) (CVPR 2022).

## Demo

https://github.com/user-attachments/assets/demo.webm

> Screen recording of the full visualization workflow: browsing episodes,
> exploring per-timestep attention heatmaps across layers and heads,
> navigating t-SNE embeddings with nearest-neighbor lines, and
> cross-panel brushing between the Attention View and Embedding View.
>
> See [assets/demo.webm](assets/demo.webm) for the source file.

## Three-Tier Architecture

```
extraction/  (Track B)    -->  data/*.h5      -->  backend/  (Track C)  -->  frontend/  (Track A)
Python + JAX + GPU             HDF5 files         FastAPI + h5py          React + TypeScript
```

1. **Extraction** runs Pi-Zero inference on VLA-Arena episodes, captures attention
   weights and Q-projections via runtime monkey-patching, pre-computes t-SNE and
   nearest neighbors, serializes to HDF5.

2. **Backend** reads HDF5 files and serves structured JSON via REST API. Zero ML
   dependencies — pure reads + numpy slicing.

3. **Frontend** renders attention heatmaps (language + camera overlays) and t-SNE
   scatter plots with cross-panel brushing. Visual theme aligned with
   [CMU Brand Standards](https://brand.cmu.edu/visual-identity/colors).

## Quick Start

```bash
# 1. Extract data (needs GPU, ~10 min for 3 episodes)
cd <openpi_root>
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  .venv/bin/python src/openpi_interpret/extraction/extract_interpret_data.py \
    --max-episodes 3 --timestep-stride 10

# 2. Start backend
cd src/openpi_interpret/backend
INTERPRET_DATA_DIR=../data conda run -n openpi-vla-arena \
  uvicorn app.main:app --host 0.0.0.0 --port 8080

# 3. Start frontend (in another terminal)
cd src/openpi_interpret/frontend
VITE_API_BASE=http://<LAN_IP>:8080/api \
  conda run -n openpi-vla-arena npx vite --host 0.0.0.0 --port 5173
```

## Vibe Operation

This project is set up for **agent-assisted operation**. An AI agent with access
to this workspace can perform all operational tasks using pre-built skills.

### Available Skills

| What you want to do | Just ask the agent |
|---------------------|-------------------|
| Launch the full visualization service | *"Launch the full service for my team"* |
| Extract data from more episodes | *"Extract 10 episodes with stride 10"* |
| Start only the backend | *"Start the backend server"* |
| Start only the frontend | *"Start the frontend dev server"* |
| Run the test suite | *"Run the backend tests"* |
| Compute CMF + Silhouette metrics | *"Run analytics on the extracted data"* |
| Run the analytics tests | *"Run the analytics test suite"* |
| Shut everything down | *"Shutdown the service"* |
| Check if services are healthy | *"Check the service health"* |

### How It Works

Skills are stored in `.cursor/skills/` and are automatically discovered by the
agent. Each skill contains step-by-step operational recipes:

| Skill | File | Purpose |
|-------|------|---------|
| `extract-model-states` | `.cursor/skills/extract-model-states/SKILL.md` | GPU extraction pipeline |
| `start-backend` | `.cursor/skills/start-backend/SKILL.md` | FastAPI on port 8080 |
| `start-frontend` | `.cursor/skills/start-frontend/SKILL.md` | Vite on port 5173 |
| `launch-full-service` | `.cursor/skills/launch-full-service/SKILL.md` | Both services for LAN |
| `run-backend-tests` | `.cursor/skills/run-backend-tests/SKILL.md` | 33 pytest tests |
| `run-analytics` | `.cursor/skills/run-analytics/SKILL.md` | CMF + Silhouette metrics |
| `run-analytics-tests` | `.cursor/skills/run-analytics-tests/SKILL.md` | 24 pytest tests |

### Agent Roles

The `.cursor/rules/` directory configures 5 specialist agent roles:

| Role | Rule File | Scope |
|------|-----------|-------|
| ML/Data Engineer | `ml-data-engineer.mdc` | Track B extraction code |
| Backend Engineer | `backend-engineer.mdc` | Track C FastAPI code |
| Frontend Engineer | `frontend-engineer.mdc` | Track A React code |
| Integration & QA Lead | `integration-qa-lead.mdc` | Cross-track validation |
| Operations Specialist | `operations-specialist.mdc` | Service deployment & ops |

Plus 4 engineering standards rules that apply automatically:
`engineering-standards.mdc`, `python-conventions.mdc`,
`typescript-react-conventions.mdc`, `testing-standards.mdc`.

### Manual Operation Reference

If you prefer to operate manually without an agent:

**Launch full service:**
```bash
fuser -k 8080/tcp 5173/tcp 2>/dev/null; sleep 1
LAN_IP=$(hostname -I | awk '{print $1}')

cd src/openpi_interpret/backend
INTERPRET_DATA_DIR=../data conda run -n openpi-vla-arena \
  uvicorn app.main:app --host 0.0.0.0 --port 8080 &
sleep 3

cd ../frontend
VITE_API_BASE="http://${LAN_IP}:8080/api" \
  conda run -n openpi-vla-arena npx vite --host 0.0.0.0 --port 5173 &
```

**Shutdown:**
```bash
fuser -k 8080/tcp 5173/tcp
```

**Health check:**
```bash
curl -s http://localhost:8080/api/health          # Backend
curl -s http://localhost:8080/api/episodes         # Data loaded?
curl -s http://localhost:5173 | head -1            # Frontend
```

### Environment Prerequisites

| Dependency | How to get it |
|-----------|---------------|
| conda env `openpi-vla-arena` | `conda create -n openpi-vla-arena python=3.11` |
| FastAPI + uvicorn | `conda activate openpi-vla-arena && pip install fastapi uvicorn[standard] pydantic-settings` |
| h5py + numpy + Pillow | Already in conda env |
| Node.js 20 | `conda install -n openpi-vla-arena nodejs=20 -c conda-forge` |
| OpenPI uv venv | `cd <openpi_root> && uv sync` |
| scikit-learn (extraction) | `uv pip install scikit-learn` |
| Pi-Zero checkpoint | Must be cached in HF hub (see `extract-model-states` skill) |
| VLA-Arena L0 S dataset | Must be cached in HF hub |

## Design Documentation

| Document | Path |
|----------|------|
| Solution design (battle-tested) | `docs/solution_design_v3/README.md` |
| Pi-Zero architecture reference | `docs/reference/pi-zero/model_architecture.md` |
| Engineering standards | `.cursor/rules/*.mdc` |
| Operational skills | `.cursor/skills/*/SKILL.md` |
