# Track A — Frontend Development (Solution Design v3)

Epic for the **OpenPI InterpreT** React application: an interactive SPA that visualizes Pi-Zero cross-attention and t-SNE token embeddings by consuming the backend REST API (Contract 2).

**Parent doc**: [Solution Design v3](../README.md)

## Purpose

Build a **React + TypeScript single-page application** that:

- Lists episodes and navigates into per-episode exploration.
- Fetches all data over **HTTP JSON** (and PNG camera URLs). **No binary loaders, no ML, no WebAssembly inference** — visualization only.

## Architecture

Data flows in one direction:

```
REST API  →  typed API client  →  custom hooks  →  container components  →  presentational components
```

- **API client** (`src/api/client.ts`): thin typed wrappers around `fetch`; central place for base URL, query strings, and error handling.
- **Custom hooks** (`useEpisodeMeta`, `useAttention`, `useTsne`, `useNeighbors`, …): encapsulate loading/error/cache concerns and map API data to view models where helpful.
- **Components**: **container/presentational split** — smart wrappers own hooks and context; dumb components receive props and render UI.

### Shared exploration state (`EpisodeContext`)

Global controls and cross-panel brushing share one React context:

| State | Role |
|-------|------|
| `timestep` | Current frame index (robot and cameras change every step). |
| `layer` | One of the sampled backbone layers. |
| `head` | Attention head index `0 … NUM_HEADS - 1`. |
| `actionIndex` | Which of the 50 action queries is selected (`0 … ACTION_HORIZON - 1`). |
| `highlightedTokenIndex` | Token index `0 … 866` for hover/brushing across Attention and t-SNE panels (or `null` when cleared). |

Implement **setters** (or updater callbacks) alongside each field so child controls and visualizations can stay decoupled.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Build | **Vite** |
| UI | **React 18** |
| Language | **TypeScript** (strict mode) |
| Styling | **Tailwind CSS** |
| Routing | **react-router** v6 |

## Token / Modality Color Map

Use the same hex palette as the backend so legends and scatter colors match JSON `color` fields where applicable:

| Key / modality | Color | Hex |
|----------------|-------|-----|
| `base_0_rgb` | Blue | `#3B82F6` |
| `left_wrist_0_rgb` | Cyan | `#06B6D4` |
| `right_wrist_0_rgb` | Teal | `#14B8A6` |
| `language` | Orange | `#F97316` |
| `state` | Green | `#22C55E` |
| `action` | Red | `#EF4444` |
| (fallback / unknown) | Gray | `#9CA3AF` |

**Schema note**: Token `type` from the API is **`"image_patch"`**, not `"image"`. Frontend conditionals and tests must use **`image_patch`** to match backend and HDF5 token meta.

## Task Breakdown

| ID | Task | Prompt |
|----|------|--------|
| **A1** | API client scaffold, types, MSW, routing | [A1-api-client-scaffold/prompt.md](./A1-api-client-scaffold/prompt.md) |
| **A2** | Episode context, home/episode shell, shared controls | [A2-navigation-shell/prompt.md](./A2-navigation-shell/prompt.md) |
| **A3** | Attention panel (summary bar, language, images, state) | [A3-attention-view/prompt.md](./A3-attention-view/prompt.md) |
| **A4** | t-SNE scatter, legend, tooltip, neighbors | [A4-embedding-view/prompt.md](./A4-embedding-view/prompt.md) |
| **A5** | Cross-panel brushing, popups, polish, live backend | [A5-polish/prompt.md](./A5-polish/prompt.md) |

## Known Pitfalls

### Pitfall #11 — Camera images are per timestep

The robot moves between frames. **`getCameraImageUrl(episodeId, camera, timestep)` must include the `timestep` query parameter** (`?timestep=N`) whenever displaying a camera frame for the current exploration step. Omitting it can show a stale or wrong frame.

### Pitfall #11 (continued) — t-SNE image patch popup

**`ImagePatchPopup`** (opened from the t-SNE tooltip for `image_patch` tokens) must receive a **`timestep` prop** and pass it through to `getCameraImageUrl`. **`TsneTooltip`** (or any intermediate wrapper) must forward **`timestep`** so the thumbnail matches the selected timestep.

### Pitfall #13 — t-SNE dark token contrast

The CMU brand token colors (Blue Thread `#043673`, Hornbostel Teal `#1F4C4C`, etc.) are predominantly dark. On a transparent or dark SVG background, these tokens are nearly invisible.

**Fix**: The t-SNE scatter SVG must have a **white background** (`backgroundColor: '#FFFFFF'`). Selected/highlighted strokes must be **black** (`#000000`), NOT white. Neighbor rings: black. Neighbor dashed lines: Iron Gray at 60% opacity.

### Token type string

Backend and HDF5 use **`type: "image_patch"`**. Do not branch on `"image"`; it will not match the schema.

### CMU Visual Theme

The UI follows [CMU Brand Standards](https://brand.cmu.edu/visual-identity/colors):
- **Headers**: Carnegie Red `#C41230` (brand hero color dominates attention)
- **Attention heatmaps**: Carnegie Red ramp `rgba(196,18,48,alpha)` — NOT Tailwind red-500
- **Token colors**: 6 colors from CMU Tartan + Campus palettes (see root README Token Map)
- **Summary bar**: Language segment (Gold Thread) needs dark text since the background is light
- Configure all CMU colors in `tailwind.config.js` under `theme.extend.colors.cmu.*`

## File Placement (reference)

All prompts in this folder describe work targeting the **`openpi_interpret/frontend/`** application root (sibling of `backend/` and `extraction/`), unless your repo pins the SPA elsewhere — keep **one** canonical frontend package and align paths in prompts with it.

Typical layout after A1–A5:

```
frontend/
  index.html
  vite.config.ts
  tailwind.config.js
  postcss.config.js
  src/
    main.tsx
    App.tsx
    api/client.ts
    types/api.ts
    types/constants.ts
    mocks/              # MSW handlers + browser/server setup
    context/EpisodeContext.tsx
    hooks/useEpisodeMeta.ts
    hooks/useAttention.ts
    hooks/useTsne.ts
    hooks/useNeighbors.ts
    pages/HomePage.tsx
    pages/EpisodePage.tsx
    components/
      layout/...
      attention/...
      embedding/...
      common/...
```

## Dependencies Between Tasks

```
A1 ──► A2 ──► A3 ──┐
              └──► A5
         A2 ──► A4 ──┘
```

A3 and A4 can proceed in parallel after A2; A5 integrates brushing and production configuration across both panels.
