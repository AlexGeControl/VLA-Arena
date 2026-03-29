# Epic: Frontend Development

> **Track A** of the OpenPI InterpreT parallel implementation plan.

This epic builds the React TypeScript SPA that visualizes Pi-Zero's attention weights and t-SNE embeddings. It communicates with the FastAPI backend (Track C) via REST API — the frontend never loads binary files or performs ML computations.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  FastAPI Backend (Track C)                                │
│  http://localhost:8080/api/                               │
│    /episodes, /attention, /tsne, /camera                  │
└────────────────────┬─────────────────────────────────────┘
                     │ REST API (JSON + image/png)
                     ▼
┌──────────────────────────────────────────────────────────┐
│  React TypeScript SPA (Vite)                              │
│                                                           │
│  ┌──────────┐   ┌──────────────────────────────────────┐ │
│  │  Home    │──▶│         Episode Explorer              │ │
│  │  Page    │   │                                       │ │
│  └──────────┘   │  ┌───────────────────────────────────┐│ │
│                  │  │  Timestamp Slider                 ││ │
│                  │  │  Shared Controls (L / H / A)      ││ │
│                  │  └──────────────┬────────────────────┘│ │
│                  │  ┌──────────────┴────────────────────┐│ │
│                  │  │  Attention   │   Embedding         ││ │
│                  │  │  View        │   View (t-SNE)      ││ │
│                  │  └──────────────┴────────────────────┘│ │
│                  └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## Key Design Decisions

- **API-driven**: All data comes from REST endpoints. No binary loaders, no client-side tensor slicing.
- **Shared controls**: Layer, head, and action token are global in `EpisodeContext`.
- **Camera images via URL**: `<img src={api.getCameraImageUrl(...)}>` — no base64.
- **Pre-computed server-side**: Attention breakdowns, t-SNE points, and nearest neighbors are all computed by the backend (or pre-computed during extraction). The frontend just renders.
- **Cross-panel brushing**: Hovering a token in one panel highlights it in the other (Task A5).

## Tasks

| Task | Deliverable | Folder | Dependency |
|------|------------|--------|------------|
| [A1. API Client & Scaffold](A1-api-client-scaffold/prompt.md) | TypeScript interfaces, API client, project skeleton, mock setup | `A1-api-client-scaffold/` | API contract (from Track C design) |
| [A2. Navigation Shell](A2-navigation-shell/prompt.md) | Home page, episode page, shared controls, timestamp slider | `A2-navigation-shell/` | Task A1 |
| [A3. Attention View](A3-attention-view/prompt.md) | Language heatmap, image overlay, state bar, summary bar | `A3-attention-view/` | Task A2 |
| [A4. Embedding View](A4-embedding-view/prompt.md) | t-SNE scatter, nearest-neighbor lines, hover tooltips | `A4-embedding-view/` | Task A2 |
| [A5. Polish & Integration](A5-polish/prompt.md) | Cross-panel brushing, contextual pop-ups, responsive layout | `A5-polish/` | Tasks A3 & A4 |

```
Task A1 ──► Task A2 ──► Task A3 ──┐
                    └──► Task A4 ──┤──► Task A5
```

Tasks A3 and A4 can be developed in parallel after Task A2.

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Build tool | Vite | Fast HMR, native TypeScript |
| UI framework | React 18 + TypeScript | Team familiarity |
| Routing | react-router v6 | SPA navigation |
| Styling | Tailwind CSS | Rapid prototyping |
| Attention heatmap (images) | HTML Canvas | Direct pixel control |
| Attention heatmap (language) | Inline styled `<span>` | Per-token coloring |
| t-SNE scatter | SVG (visx or plain React SVG) | Lightweight, DOM tooltips |
| API communication | `fetch` + typed client | No heavy HTTP library needed |
| Mock API (dev) | MSW (Mock Service Worker) | Intercepts fetch, returns fixtures |

## Token Map

| Token Range | Count | Modality | Color |
|-------------|-------|----------|-------|
| 0–255 | 256 | Base camera patches | Blue `#3B82F6` |
| 256–511 | 256 | Left wrist patches | Cyan `#06B6D4` |
| 512–767 | 256 | Right wrist patches | Teal `#14B8A6` |
| 768–815 | ≤48 | Language tokens | Orange `#F97316` |
| 816 | 1 | State | Green `#22C55E` |
| 817–866 | 50 | Action tokens | Red `#EF4444` |

## File Placement

```
openpi_interpret/
  frontend/
    src/
      api/                # API client
      components/         # React components
      context/            # EpisodeContext
      hooks/              # Data fetching hooks
      types/              # TypeScript interfaces + constants
```
