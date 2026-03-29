# Task A3: Attention View

> Part of [Frontend Development (Track A)](../README.md). **Depends on [A2](../A2-navigation-shell/prompt.md).**

## Goal

Replace the **Attention** placeholder with a full **action-centric attention** panel: summary distribution, **language** inline heatmap, **per-camera patch** grids with bilinear smoothing, and **state** bar — all driven by **`api.getAttention`** and **`EpisodeContext`**.

## Task

### 1. `useAttention` hook

**`useAttention(episodeId: string)`** (or pass ids from context):

- Reads **`timestep`**, **`layer`**, **`head`**, **`actionIndex`** from **`useEpisode()`**.
- Calls **`api.getAttention(episodeId, timestep, layer, head, actionIndex)`** when inputs change.
- Returns `{ data, loading, error }` for **`AttentionResponse`**.

Optional: also fetch **`getAttentionSummary`** in the same hook or a sibling if the summary bar uses aggregated data; if the stacked bar is derived purely from `breakdown` for the selected action, one request is enough. Prefer **one network round-trip** unless UX requires summary-for-all-actions (then use **`getAttentionSummary`**).

### 2. `AttentionSummaryBar`

- **Stacked horizontal bar** (or vertical segments) showing approximate **share** of attention mass across **images / language / state / actions** for the **current query** (selected action).
- Use **`breakdown`** from `AttentionResponse`: combine camera totals into **images**, use **`language_total`**, **`state_weight`**, **`action_total`** (or equivalent normalization from `row`).
- Display **percent labels** or tooltips; ensure segments sum visually to ~100% (normalize if raw totals are not exactly 1 due to float noise).

### 3. `LanguageAttention`

- Render the instruction as **inline spans** (one per language token position) using metadata from **`getTokenMeta`** or weights aligned with `breakdown.language_weights`.
- **Color ramp**: white → red (low → high weight) **within the language row** (local normalization: divide by max language weight for that query).
- **Hover**: call context setter to update **`highlightedTokenIndex`** for the token index corresponding to that span (for cross-panel sync in A5).

### 4. `ImageAttention` (per camera)

For each camera in **`CAMERA_NAMES`** (or episode meta):

- **Background**: `<img>` whose **`src`** is **`api.getCameraImageUrl(episodeId, cam, timestep)`**.
  - **MUST pass `timestep`** — Pitfall #11: robot moves; omitting timestep shows wrong frames.
- **Overlay**: HTML **`canvas`** (e.g. **16×16** cells matching ViT-style patch grid) drawn on top of the image (position absolute, object-fit aligned with image).
- **Weights**: map **`breakdown.cameras[cam]`** (length 256) to a **16×16** heatmap. Use **bilinear interpolation** (or upsample with smooth gradients) so cells are not harsh rectangles only — spec: “**16×16 canvas overlay with bilinear interpolation**” for the heatmap sampling/upsample step.
- **Hover**: `onHover` reports a **global token index** using **`cameraOffset + row * 16 + col`** (define **`cameraOffset`** consistently with backend token order — document the mapping in a code comment next to constants; must match HDF5 token layout).
- **Highlight**: when **`highlightedTokenIndex`** from context equals a patch index for this camera, draw a **red rectangle** outline on that cell.
- Forward hover events to **`setHighlightedTokenIndex`** (and clear on mouse leave as appropriate).

### 5. `StateAttention`

- Single **horizontal bar** (or thin heat strip) for **`state_weight`** relative to other modalities or normalized locally — keep consistent with summary bar semantics.
- Hover sets **`highlightedTokenIndex`** for the **state** token index (fixed index in the 867 layout — document constant).

### 6. Composition

**`AttentionView`** container:

- Uses **`useAttention`** + **`useEpisode`** + **`useEpisodeMeta`** / **`getTokenMeta`** as needed.
- Renders **`AttentionSummaryBar`**, **`LanguageAttention`**, three **`ImageAttention`** blocks, **`StateAttention`**.
- Loading skeletons and error banner.

## Acceptance Criteria

1. Changing **timestep**, **layer**, **head**, or **action** refetches attention and updates all subviews.
2. **`getCameraImageUrl`** is always called with **`timestep`** for episode views.
3. Language weights use **local** normalization (white–red) and are readable on dark/light background (pick contrast appropriately).
4. Image overlay is **16×16** logical grid with **smooth** (bilinear) appearance, not only raw nearest-neighbor blocks.
5. Hovering language or image patches updates **`highlightedTokenIndex`** in context (value matches global token index scheme).
6. **`highlightedTokenIndex`** draws a **red rectangle** on the corresponding image patch when set from elsewhere (minimal test: temporary button or simulate from devtools — full cross-panel story completed in A5).

## Out of Scope

- t-SNE panel (A4), tooltip popups for language/patch (A5), final visual polish fonts (A5).
