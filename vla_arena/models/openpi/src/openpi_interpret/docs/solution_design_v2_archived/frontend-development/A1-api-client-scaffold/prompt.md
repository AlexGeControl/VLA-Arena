# Task A1: API Client & Project Scaffold

## Context

We are building **OpenPI InterpreT**, a React TypeScript SPA that visualizes the internal representations of the Pi-Zero VLA model. The frontend communicates with a **FastAPI backend** (Track C) via REST API — it never loads binary files or parses raw tensors.

This is Task A1 of Track A (Frontend) — we set up the project skeleton, define TypeScript interfaces matching the API response schemas, and create an API client layer.

**Key domain facts** (from [model_architecture.md](../../../../../../docs/pi-zero/model_architecture.md)):
- Pi-Zero's backbone samples **7 layers** at stride 3: `[0, 3, 6, 9, 12, 15, 17]`.
- Each sampled layer has **8 attention heads**.
- The unified token sequence has **~867 tokens**: 768 image patches + up to 48 language + 1 state + 50 action.
- We store only **suffix queries** (51 tokens) attending to all 867 keys.

**Data source**: All data is served by the FastAPI backend at `http://localhost:8080/api/`. The frontend makes `fetch` calls to REST endpoints and receives JSON responses.

## Task

### 1. Initialize the project

Create a Vite + React 18 + TypeScript project at:

```
openpi_interpret/frontend/
```

Install dependencies:
- `react`, `react-dom`, `react-router-dom` (v6)
- `tailwindcss` (v3), `postcss`, `autoprefixer`
- TypeScript, with strict mode enabled

### 2. Define TypeScript interfaces

Create `src/types/api.ts` matching the backend's Pydantic schemas:

```typescript
export const SAMPLED_LAYERS = [0, 3, 6, 9, 12, 15, 17] as const;
export type SampledLayer = (typeof SAMPLED_LAYERS)[number];

export type CameraName = "base_0_rgb" | "left_wrist_0_rgb" | "right_wrist_0_rgb";

export interface EpisodeSummary {
  episode_id: string;
  task_instruction: string;
  num_timesteps: number;
}

export interface EpisodeMeta {
  episode_id: string;
  task_instruction: string;
  num_timesteps: number;
  instruction_tokens: string[];
  sampled_layers: SampledLayer[];
  camera_names: CameraName[];
}

export interface TokenMeta {
  index: number;
  type: "image_patch" | "language" | "state" | "action";
  source: string;
  patch_row?: number;
  patch_col?: number;
  token_text?: string;
  token_position?: number;
}

export interface AttentionBreakdown {
  cameras: Record<CameraName, number[]>;
  camera_totals: Record<CameraName, number>;
  language_weights: number[];
  language_total: number;
  state_weight: number;
  action_weights: number[];
  action_total: number;
}

export interface AttentionResponse {
  row: number[];
  breakdown: AttentionBreakdown;
}

export interface AttentionSummary {
  modality_totals: {
    images: number;
    language: number;
    state: number;
    actions: number;
  };
  per_action: number[];
}

export interface TsnePoint {
  index: number;
  x: number;
  y: number;
  type: string;
  source: string;
  color: string;
}

export interface TsneResponse {
  points: TsnePoint[];
}

export interface NearestNeighbor {
  index: number;
  x: number;
  y: number;
  distance: number;
  modality_group: string;
  type: string;
  source: string;
}

export interface NeighborResponse {
  selected: { index: number; x: number; y: number };
  neighbors: NearestNeighbor[];
}
```

### 3. Create API client

Create `src/api/client.ts`:

```typescript
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080/api";

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`API error ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

export const api = {
  listEpisodes: () =>
    get<EpisodeSummary[]>("/episodes"),

  getEpisode: (id: string) =>
    get<EpisodeMeta>(`/episodes/${id}`),

  getCameraImageUrl: (id: string, camera: CameraName) =>
    `${API_BASE}/episodes/${id}/camera/${camera}`,

  getTokenMeta: (id: string, timestep: number) =>
    get<TokenMeta[]>(`/episodes/${id}/timesteps/${timestep}/token-meta`),

  getAttention: (id: string, timestep: number, layer: SampledLayer, head: number, action: number) =>
    get<AttentionResponse>(
      `/episodes/${id}/timesteps/${timestep}/attention?layer=${layer}&head=${head}&action=${action}`
    ),

  getAttentionSummary: (id: string, timestep: number, layer: SampledLayer, head: number) =>
    get<AttentionSummary>(
      `/episodes/${id}/timesteps/${timestep}/attention/summary?layer=${layer}&head=${head}`
    ),

  getTsne: (id: string, timestep: number, layer: SampledLayer) =>
    get<TsneResponse>(`/episodes/${id}/timesteps/${timestep}/tsne?layer=${layer}`),

  getNeighbors: (id: string, timestep: number, layer: SampledLayer, action: number) =>
    get<NeighborResponse>(
      `/episodes/${id}/timesteps/${timestep}/tsne/neighbors?layer=${layer}&action=${action}`
    ),
};
```

### 4. Create helper constants

Create `src/types/constants.ts`:

```typescript
// SAMPLED_LAYERS is defined in api.ts — import from there, do not duplicate.
export const NUM_HEADS = 8;
export const ACTION_HORIZON = 50;
export const PATCH_GRID_SIZE = 16;
export const PATCHES_PER_CAMERA = 256;
export const CAMERA_NAMES = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"] as const;

export const TOKEN_COLORS: Record<string, string> = {
  base_0_rgb: "#3B82F6",
  left_wrist_0_rgb: "#06B6D4",
  right_wrist_0_rgb: "#14B8A6",
  language: "#F97316",
  state: "#22C55E",
  action: "#EF4444",
};
```

### 5. Development without backend

For frontend development before the backend is ready, configure Vite to proxy API requests to a mock server, or use a static JSON fixture:

- **Option A**: Use [MSW (Mock Service Worker)](https://mswjs.io/) to intercept `fetch` calls and return fixture data.
- **Option B**: Create a small Express/json-server that serves static JSON files matching the API schema.
- **Option C**: Start the backend with test HDF5 files (preferred once Track C delivers C1-C3).

The recommended approach is **Option A** (MSW) for early Task A1–A2 development, switching to the real backend once Track C delivers.

### 6. Set up routing shell

Create minimal routing in `src/App.tsx`:
- `/` → `HomePage` (placeholder: "OpenPI InterpreT — Home")
- `/episode/:episodeId` → `EpisodePage` (placeholder: "Episode: {episodeId}")

## Acceptance Criteria

- [ ] `npm run dev` starts the Vite dev server without errors
- [ ] Navigating to `/` shows the Home placeholder
- [ ] Navigating to `/episode/mock_001` shows the Episode placeholder
- [ ] `api.listEpisodes()` returns data (from mock or real backend)
- [ ] All TypeScript interfaces compile with strict mode, no `any` types
- [ ] `VITE_API_BASE` environment variable overrides the API base URL
