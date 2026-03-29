# Task A4: Embedding View (t-SNE)

> Part of [Frontend Development (Track A)](../README.md). **Depends on [A2](../A2-navigation-shell/prompt.md).** Can run in parallel with A3 after A2.

## Goal

Implement the **t-SNE scatter** panel: **867** points, **modality coloring**, **selected action** emphasis, **five dashed neighbor lines**, legend, and a **tooltip** with contextual detail — including **language** and **image patch** popups. Wiring to cross-panel brushing is finalized in A5, but the tooltip must already accept the data needed for Pitfall #11.

## Task

### 1. `useTsne` hook

- Inputs: **`episodeId`**, and from context **`timestep`**, **`layer`**.
- Calls **`api.getTsne(episodeId, timestep, layer)`**.
- Returns `{ data, loading, error }`.

### 2. `useNeighbors` hook

- Inputs: **`episodeId`**, **`timestep`**, **`layer`**, **`actionIndex`** from context.
- Calls **`api.getNeighbors(episodeId, timestep, layer, actionIndex)`**.
- Returns `{ data, loading, error }`.
- Skip or cancel request if `actionIndex` out of range.

### 3. `TsneScatterPlot` (SVG preferred)

- Plot **867 circles** from **`TsneResponse.points`**.
- **Axes**: normalize `x`/`y` to SVG viewBox coordinates with modest padding; handle aspect ratio.
- **Fill**: use each point’s **`color`** from API when present; else fall back to **`TOKEN_COLORS`** via `type` / `source` (remember **`type === "image_patch"`** for patches).
- **Selected action token**: the point whose **`index`** matches the **suffix action token index** for current **`actionIndex`** — larger radius + **white stroke ring** (or outer glow) per design.
- **Neighbors**: when **`NeighborResponse`** loads, draw **5** **dashed** lines from **`selected`** to each neighbor point `(x, y)`.
- **Hover**: mouse over circle updates **`highlightedTokenIndex`** in context (A5 will mirror attention panel).

### 4. `TsneLegend`

- Show **6** modality entries matching the color map: three cameras, language, state, action (see parent [README](../README.md) token table).
- Small swatch + label; compact placement (corner overlay or below plot).

### 5. `TsneTooltip`

- On hover, show floating tooltip near cursor (portal optional).
- Content rules:
  - **Common**: token `index`, `type`, `source`, `x`/`y` rounded.
  - **`language`**: delegate to **`LanguageTokenPopup`**-lite inline or expandable snippet showing **full instruction** with hovered token emphasized (full popup polish in A5).
  - **`image_patch`**: open **`ImagePatchPopup`** (or inline thumbnail) showing the **128×128** patch preview — see below.

**Pitfall #11 — `ImagePatchPopup` and timestep**

- **`ImagePatchPopup`** MUST receive **`timestep: number`** (and `episodeId`, `camera`/`source`, `patch_row`, `patch_col` as needed).
- **`TsneTooltip`** MUST pass **`timestep`** through — do not read timestep only from a stale closure or omit it; the camera URL must use **`api.getCameraImageUrl(episodeId, camera, timestep)`**.

### 6. `EmbeddingView` container

- Compose **`TsneScatterPlot`**, **`TsneLegend`**, **`TsneTooltip`**.
- Loading / error states.

## Acceptance Criteria

1. **867** points render without measurable lag on a laptop (use single SVG layer or virtualized strategy if needed; document if deviating).
2. **Selected action** point is visually distinct (**larger + white ring**).
3. **Five neighbor lines** appear dashed from selected action to neighbors when neighbors API succeeds.
4. **Legend** shows **6** colors with correct labels.
5. Tooltip appears on hover with correct metadata; for **`image_patch`**, thumbnail uses **`getCameraImageUrl(..., timestep)`** with **`timestep` prop** threaded through **`TsneTooltip`** → **`ImagePatchPopup`**.
6. Token **`type`** checks use **`"image_patch"`**, not `"image"`.

## Out of Scope

- Full **cross-panel** hover sync from attention → t-SNE (A5).
- MSW disable / production env (A5).
