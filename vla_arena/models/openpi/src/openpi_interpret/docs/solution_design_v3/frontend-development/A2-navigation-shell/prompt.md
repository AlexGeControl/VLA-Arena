# Task A2: Navigation Shell & Shared Controls

> Part of [Frontend Development (Track A)](../README.md). **Depends on [A1](../A1-api-client-scaffold/prompt.md).**

## Goal

Introduce **shared exploration state** via `EpisodeContext`, wire **episode list → episode detail** navigation, and lay out the two-panel grid with **timestep** and global controls. Attention and t-SNE views remain **placeholders** until A3/A4.

## Task

### 1. `EpisodeContext` + provider

Create `src/context/EpisodeContext.tsx` (or equivalent):

**State (with setters):**

| Field | Type | Initial / notes |
|-------|------|-----------------|
| `timestep` | `number` | `0` |
| `layer` | `SampledLayer` | e.g. first value of `SAMPLED_LAYERS` |
| `head` | `number` | `0` … `NUM_HEADS - 1` |
| `actionIndex` | `number` | `0` … `ACTION_HORIZON - 1` |
| `highlightedTokenIndex` | `number \| null` | `null` when nothing hovered |

Export:

- `EpisodeProvider` wrapping children (accepts `episodeId: string` prop if you need it for hooks).
- `useEpisode()` hook throwing if used outside provider.

Keep the context **focused** on exploration UI state — do not store full API responses here (those belong in query hooks or component state).

### 2. `useEpisodeMeta` hook

- **`useEpisodeMeta(episodeId: string)`** calls **`api.getEpisode(episodeId)`**.
- Returns `{ data, loading, error, refetch }` (or React Query if you adopt it later — plain `useEffect` + `useState` is enough for v3).
- Used on the episode page to populate instruction text, validate `timestep` bounds against `num_timesteps`, and populate layer dropdown options from `sampled_layers` if the API ever diverges from constants (prefer API when present).

### 3. `HomePage`

- On mount, call **`api.listEpisodes()`**.
- Render **cards** (or a simple list): episode id, truncated instruction, timestep count.
- Clicking a card **`navigate(`/episode/${episode_id}`)`**.
- Loading and error states visible (spinner / message).

### 4. `EpisodePage`

- Read **`episodeId`** from `useParams()`.
- Wrap content in **`EpisodeProvider`** (pass `episodeId` if needed).
- **`useEpisodeMeta(episodeId)`** at page level; gate render on success or show error UI.
- **`TimestepSlider`**: range `0 … num_timesteps - 1`, bound to **`timestep`** in context. Stepping updates **`timestep`** only (resets `highlightedTokenIndex` optional but nice).
- **`SharedControls`** row:
  - **Layer** dropdown: only **sampled** values (`SAMPLED_LAYERS` or meta’s `sampled_layers`).
  - **Head** dropdown: **`0`–`7`** (`NUM_HEADS`).
  - **Action** dropdown: **`0`–`49`** (`ACTION_HORIZON`).
- **Two-panel grid**:
  - Left: placeholder **`AttentionView`** (“Attention — A3”).
  - Right: placeholder **`EmbeddingView`** (“t-SNE — A4”).
- Responsive: **`grid-cols-1 lg:grid-cols-2`**, gap and min-heights so panels stack on small screens.

### 5. Placeholder components

- **`AttentionView`**: bordered region, title, short note that A3 will render heatmaps and bars.
- **`EmbeddingView`**: bordered region, title, note that A4 will render scatter + neighbors.

No real charts yet.

## Acceptance Criteria

1. Navigating `/` lists episodes from the API (or MSW); clicking opens `/episode/:id`.
2. Invalid or unknown `episodeId` shows a clear error (404 from API or empty list handling).
3. **`EpisodeProvider`** supplies all five state fields + setters; **`useEpisode()`** works on nested components.
4. **Timestep**, **layer**, **head**, and **action** controls update context and are visible on the episode page.
5. Layout is **two columns on `lg+`**, **single column** below breakpoint.
6. No attention/t-SNE API calls required yet (placeholders only).

## Out of Scope

- Attention visualization (A3), t-SNE (A4), cross-panel brushing (A5).
