# Task A4: Embedding View (t-SNE)

## Context

Tasks A1–A2 created the project scaffold and navigation shell with shared controls. The Embedding View panel currently shows a placeholder. This phase implements the t-SNE scatter plot visualization.

**What we're visualizing**: Pre-computed 2D t-SNE projections of all ~867 token embeddings at a selected layer. Each point is a token, colored by modality. Given a selected action token, we highlight its nearest neighbor in each modality group.

**Embedding space**: t-SNE coordinates are computed from **head-space Q-projections** (8 heads × 256 dims = 2048 concatenated) during the extraction phase. The backend serves pre-computed coordinates and neighbors.

**Data source**: Two API endpoints:
- `GET .../tsne?layer=L` → all 867 points with metadata and colors
- `GET .../tsne/neighbors?layer=L&action=A` → 5 nearest neighbors (one per modality group)

No client-side distance computation or t-SNE indexing needed.

**Prerequisite**: Task A2 is complete. The Embedding View reads `layer` and `actionIndex` from the shared `EpisodeContext`.

## Task

### 1. Data loading hooks

Create `src/hooks/useTsne.ts`:
- Calls `api.getTsne(episodeId, timestep, layer)` when episodeId, timestep, or layer changes
- Returns `TsneResponse` (867 points with x, y, type, source, color)

Create `src/hooks/useNeighbors.ts`:
- Calls `api.getNeighbors(episodeId, timestep, layer, actionIndex)` when any selection changes
- Returns `NeighborResponse` (selected point + 5 neighbors)

### 2. Scatter plot

Render an SVG-based scatter plot:
- SVG fills available panel width, 1:1 aspect ratio, minimum 500×500px.
- Coordinate mapping: data min/max with 10% padding → SVG viewport.
- Each token as a `<circle>`: radius 3px, fill from `point.color`, opacity 0.6.
- Selected action token: radius 6px, full opacity, white stroke ring.
- Legend in top-right: 6 modality colors with labels.

### 3. Nearest-neighbor lines

For each of the 5 neighbors from the API:
- Dashed SVG `<line>` from selected action to neighbor, colored by neighbor's modality.
- Neighbor point: 5px radius, white stroke highlight.
- Midpoint label: modality group + distance (e.g., "base_0_rgb: 3.2").

### 4. Hover tooltips

On point hover, show a tooltip with:
- Image patch: "base_0_rgb patch (row 5, col 12)"
- Language: "token: 'shelf' (position 10)"
- State: "proprioceptive state"
- Action: "action token 23"

Absolutely-positioned `<div>` following mouse, controlled via React state.

### Component structure

```
src/components/
  EmbeddingView.tsx
  embedding/
    TsneScatterPlot.tsx
    TsneLegend.tsx
    TsneTooltip.tsx
    NearestNeighborLines.tsx
src/hooks/
  useTsne.ts
  useNeighbors.ts
```

### Acceptance Criteria

- [ ] Scatter plot renders ~867 points with correct modality colors
- [ ] Selected action token visually distinct (larger, white ring)
- [ ] Changing layer re-renders with different coordinates (new API call)
- [ ] Changing action token updates highlight and triggers neighbor API call
- [ ] 5 dashed lines to nearest neighbors in each modality group
- [ ] Hover tooltip with correct token info
- [ ] Legend maps colors to modality names
- [ ] No client-side distance computation (all from API)
