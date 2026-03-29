# Task A2: Navigation Shell

## Context

Task A1 created the project scaffold, TypeScript data interfaces, API client, and mock setup (MSW). We now need to build the navigation UI that users interact with to browse episodes and explore individual timesteps. This task creates the full page layout with **shared controls** — but the two visualization panels (Attention View, Embedding View) remain empty placeholders that Tasks A3 and A4 will fill in.

**Key change from v1**: Controls are **shared globally** between both panels, not independent per-panel. Layer, action token, and head selections are managed in a single shared context. The head selector is only relevant to the Attention View but lives in the shared control bar.

**Prerequisite**: Task A1 is complete. The project compiles and serves via `npm run dev`. Mock API responses are available via MSW (or a running backend).

## Task

### 1. Home Page (`/`)

Build a home page that:
- Calls `api.listEpisodes()` on mount to fetch the episode list from the backend.
- Displays a card for each episode showing: episode ID, task instruction (truncated to 80 chars), and number of timesteps.
- Clicking a card navigates to `/episode/{episode_id}`.
- Shows a loading spinner while fetching.
- Includes a header with the title "OpenPI InterpreT" and a subtitle "Interactive Pi-Zero Attention & Embedding Explorer".

### 2. Episode Page layout (`/episode/:episodeId`)

Build the episode page with the following vertical layout:

```
┌─────────────────────────────────────────────────────────────────┐
│  Header: Episode ID + Task Instruction (full text)              │
├─────────────────────────────────────────────────────────────────┤
│  Timestamp Slider: [0] ───●─────────────── [N-1]   Step: 2     │
├─────────────────────────────────────────────────────────────────┤
│  Shared Controls:                                               │
│  Layer: [▼ 0,3,6,9,12,15,17]  Head: [▼ 0-7]  Action: [▼ 0-49] │
├───────────────────────────────┬─────────────────────────────────┤
│  Attention View               │  Embedding View                 │
│                               │                                 │
│  ┌─ Placeholder ───────────┐  │  ┌─ Placeholder ───────────┐   │
│  │                         │  │  │                         │   │
│  │  "Attention View"       │  │  │  "Embedding View"       │   │
│  │  "(Task A3)"            │  │  │  "(Task A4)"            │   │
│  │                         │  │  │                         │   │
│  └─────────────────────────┘  │  └─────────────────────────┘   │
└───────────────────────────────┴─────────────────────────────────┘
```

### 3. Data loading

- On mount, call `api.getEpisode(episodeId)` to fetch episode metadata from the backend.
- Store the `EpisodeMeta` in a React context so child components can access it without prop-drilling.
- Show a loading skeleton while the data is being fetched.
- Attention and t-SNE data are **not** loaded here — they are fetched on demand by the individual panels via their own API calls when the user selects a layer/timestep.

### 4. Timestamp slider

- A horizontal range slider from 0 to `num_timesteps - 1`.
- Displays the current timestep index numerically next to the slider.
- Changing the slider updates all downstream components (both panels read the current timestep).

### 5. Shared control bar

A single control bar below the timestamp slider, above both panels. All controls are global:

- **Layer** dropdown: options from `SAMPLED_LAYERS` (0, 3, 6, 9, 12, 15, 17), default 0.
- **Head** dropdown: options 0 through 7, default 0. Labeled "(Attention View only)" to indicate it only affects the left panel.
- **Action token** dropdown: options 0 through 49 (labeled "Action 0" ... "Action 49"), default 0.

All three selections are stored in the shared `EpisodeContext`.

### 6. Episode Context

Create `src/context/EpisodeContext.tsx` that provides:

```typescript
interface EpisodeState {
  meta: EpisodeMeta | null;
  loading: boolean;
  error: string | null;

  // Shared selections
  timestep: number;
  layer: SampledLayer;
  head: number;
  actionIndex: number;

  // Cross-panel brushing (Task A5 will use this)
  highlightedTokenIndex: number | null;

  // Setters
  setTimestep: (t: number) => void;
  setLayer: (l: SampledLayer) => void;
  setHead: (h: number) => void;
  setActionIndex: (a: number) => void;
  setHighlightedTokenIndex: (idx: number | null) => void;
}
```

### 7. Component structure

```
src/
  components/
    HomePage.tsx
    EpisodePage.tsx
    TimestepSlider.tsx
    SharedControls.tsx          # Shared control bar (layer/head/action dropdowns)
    AttentionView.tsx           # Placeholder for Task A3
    EmbeddingView.tsx           # Placeholder for Task A4
  context/
    EpisodeContext.tsx           # React context for episode data + shared selections
  hooks/
    useEpisodeMeta.ts           # Fetch episode metadata via api.getEpisode()
  api/
    client.ts                   # REST API client (from Task A1)
  types/
    api.ts                      # TypeScript interfaces (from Task A1)
    constants.ts                # Constants (from Task A1)
```

### Acceptance Criteria

- [ ] Home page lists episodes from the API, clicking navigates to the episode page
- [ ] Episode page shows header with episode ID and full task instruction
- [ ] Timestamp slider works and displays the current index
- [ ] Shared control bar has working dropdowns with correct option ranges (7 layers, 8 heads, 50 actions)
- [ ] Layer dropdown shows sampled layer indices: 0, 3, 6, 9, 12, 15, 17
- [ ] Changing any control updates the `EpisodeContext` state
- [ ] The two placeholder panels render their names and phase numbers
- [ ] Layout is responsive: on screens narrower than 1024px, panels stack vertically
- [ ] No TypeScript errors, no console warnings
