# Task A3: Attention View

## Context

Tasks A1–A2 created the project scaffold, API client, and navigation shell with shared controls. The Attention View panel currently shows a placeholder. This phase implements the actual attention visualization.

**What we're visualizing**: A single row of the attention matrix — the attention distribution of one selected **action token** (query) over all **~867 key tokens**. This tells us: *"When generating this action, what did the model look at?"*

**Data source**: The API endpoint `GET /api/episodes/{id}/timesteps/{t}/attention?layer=L&head=H&action=A` returns the full attention row and a pre-computed modality breakdown. No client-side binary parsing or slicing needed.

**Layer values**: The layer dropdown shows only sampled layers: 0, 3, 6, 9, 12, 15, 17.

**Prerequisite**: Task A2 is complete. The navigation shell works with shared dropdowns.

## Task

### 1. Attention data loading

Create a hook `src/hooks/useAttention.ts` that:
- Reads `episodeId`, `timestep`, `layer`, `head`, `actionIndex` from the shared `EpisodeContext`
- Calls `api.getAttention(episodeId, timestep, layer, head, actionIndex)` when any selection changes
- Returns the `AttentionResponse` (or null while loading)
- Manages loading/error state

### 2. Summary bar

At the top of the Attention View, render a compact horizontal **stacked bar** using the breakdown's totals:
- Images total (sum of 3 camera_totals) → blue
- `language_total` → orange
- `state_weight` → green
- `action_total` → red

Each segment's width is proportional to that group's share. Labels show percentages.

### 3. Language attention display

- Section header: **"Language Instruction"** with `language_total` as a percentage badge.
- Render instruction as inline `<span>` elements, one per token.
- Background color: white-to-red ramp based on `breakdown.language_weights[i]`, locally normalized to the max within the language group.
- Small gap between spans, thin bottom border for boundaries.

### 4. Image patch attention overlay

For each camera (from `EpisodeMeta.camera_names`):
- Sub-header: camera name + `camera_totals[cam]` as percentage badge.
- Camera image via `<img src={api.getCameraImageUrl(episodeId, cam)}>` at 256×256px.
- Canvas overlay: reshape `breakdown.cameras[cam]` (256 values) into 16×16, draw as heatmap with bilinear interpolation.
- If camera is not in `camera_names`, show dimmed "No image" placeholder.

### 5. State attention display

- Header: **"Proprioceptive State"** + `state_weight` as percentage badge.
- Horizontal progress bar proportional to `state_weight`.

### Component structure

```
src/components/
  AttentionView.tsx
  attention/
    AttentionSummaryBar.tsx
    LanguageAttention.tsx
    ImageAttention.tsx
    StateAttention.tsx
src/hooks/
  useAttention.ts
```

### Acceptance Criteria

- [ ] Selecting a different layer/head/action token triggers an API call and updates the visualization
- [ ] Language tokens colored on white-to-red scale with local normalization
- [ ] Image patch overlays show smooth heatmaps (bilinear interpolation at 256×256)
- [ ] Summary bar percentages sum to ~100%
- [ ] Masked cameras show placeholder
- [ ] No layout shift when switching timesteps
- [ ] Camera images loaded via API URL (not base64)
