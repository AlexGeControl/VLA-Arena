# Frontend SPA (Track A)

React TypeScript application for visualizing Pi-Zero's attention patterns
and token embeddings. Communicates with the FastAPI backend via REST API.

## Architecture

```
Backend REST API → api/client.ts → hooks/ → components/
```

- `src/api/client.ts` — Typed fetch wrapper for all 9 API endpoints.
- `src/hooks/` — Custom hooks for data fetching (`useAttention`, `useTsne`, `useNeighbors`, `useEpisodeMeta`).
- `src/context/EpisodeContext.tsx` — Shared state: layer, head, action, timestep, highlighted token.
- `src/components/attention/` — Attention heatmaps: language spans, image canvas overlays, summary bar.
- `src/components/embedding/` — t-SNE scatter plot, nearest-neighbor lines, tooltips, contextual pop-ups.
- `src/components/common/` — Reusable Badge and Skeleton components.

## Key Visualizations

- **Attention View**: Per-head cross-attention from action tokens to all condition tokens. White-to-red heatmap on language spans, bilinear-interpolated 16×16 canvas overlay on camera images.
- **Embedding View**: t-SNE scatter of 867 tokens colored by modality. Dashed lines to 5 nearest neighbors (one per modality group). Head selection has no effect here — t-SNE is computed from concatenated head-space Q-projections.

## Usage

```bash
# Development
VITE_API_BASE=http://192.168.3.57:8080/api \
  conda run -n openpi-vla-arena npx vite --host 0.0.0.0 --port 5173

# Production build
conda run -n openpi-vla-arena npm run build
```
