# Task A1: API Client Scaffold

> Part of [Frontend Development (Track A)](../README.md). No upstream task dependency — this is the foundation for A2–A5.

## Goal

Bootstrap a **Vite + React 18 + TypeScript (strict) + Tailwind CSS** project and implement a fully typed REST client plus **MSW** mocks so the UI can be developed before the backend is available.

## Task

### 1. Project setup

- Initialize Vite with the **React + TypeScript** template.
- Enable **strict** TypeScript (`strict`, `noUnusedLocals`, `noUnusedParameters` as appropriate).
- Add **Tailwind CSS** (PostCSS pipeline), **react-router-dom** v6, and **MSW** (dev + test).

Environment:

- **`VITE_API_BASE`**: base URL for the API, e.g. `http://localhost:8080/api`. The client should read `import.meta.env.VITE_API_BASE` and default sensibly for local dev.

### 2. `src/types/api.ts`

Define **all** of the following interfaces and types. Names and field shapes must stay aligned with the FastAPI Pydantic models (Contract 2).

- **`SAMPLED_LAYERS`**: `as const` tuple `[0, 3, 6, 9, 12, 15, 17]`.
- **`SampledLayer`**: element type of `SAMPLED_LAYERS`.
- **`CameraName`**: union matching episode camera keys, e.g. `"base_0_rgb" | "left_wrist_0_rgb" | "right_wrist_0_rgb"`.
- **`EpisodeSummary`**: `episode_id`, `task_instruction`, `num_timesteps`.
- **`EpisodeMeta`**: `episode_id`, `task_instruction`, `num_timesteps`, `instruction_tokens`, `sampled_layers`, `camera_names`.
- **`TokenMeta`**: `index`, `type` (use a string union that includes **`"image_patch"`**, `"language"`, `"state"`, `"action"`), `source`, optional `patch_row`, `patch_col`, `token_text`, `token_position`.
- **`AttentionBreakdown`**: `cameras` (per-camera float arrays), `camera_totals`, `language_weights`, `language_total`, `state_weight`, `action_weights`, `action_total`.
- **`AttentionResponse`**: `row: number[]`, `breakdown: AttentionBreakdown`.
- **`AttentionSummary`**: `modality_totals` (`images`, `language`, `state`, `actions`), `per_action: number[]`.
- **`TsnePoint`**: `index`, `x`, `y`, `type`, `source`, `color`.
- **`TsneResponse`**: `points: TsnePoint[]`.
- **`NearestNeighbor`**: `index`, `x`, `y`, `distance`, `modality_group`, `type`, `source`.
- **`NeighborResponse`**: `selected: { index, x, y }`, `neighbors: NearestNeighbor[]`.

Export **`SAMPLED_LAYERS`** and **`SampledLayer`** from this file (A2+ import layer types from here).

### 3. `src/types/constants.ts`

Domain constants **not** duplicated from the API schema:

- **`NUM_HEADS`**: `8` (must match Pi-Zero / extraction).
- **`ACTION_HORIZON`**: `50` (suffix action queries).
- **`TOKEN_COUNT`**: `867` (prefix + suffix tokens).
- **`TOKEN_COLORS`**: map from modality / camera key to hex (see parent [README token table](../README.md)).
- **`CAMERA_NAMES`**: tuple or array of `CameraName` in UI order (import `CameraName` from `api.ts` if needed).
- **`MODALITY_LABELS`** (optional): human-readable labels for legend copy.

**Import `SAMPLED_LAYERS` from `./api.ts`** (or `../types/api` per your structure) — do not redefine the layer list in two places.

### 4. `src/api/client.ts`

Implement a small **`get<T>(path: string): Promise<T>`** helper that:

- Prefixes paths with `VITE_API_BASE`.
- Sets `Accept: application/json`.
- On non-OK responses, throws an `Error` with status text or body snippet.

Export an **`api`** object with **nine** surface methods matching backend routes:

| Method | HTTP |
|--------|------|
| `health()` | `GET /api/health` → `{ status: string }` |
| `listEpisodes()` | `GET /api/episodes` → `EpisodeSummary[]` |
| `getEpisode(id)` | `GET /api/episodes/{id}` → `EpisodeMeta` |
| `getCameraImageUrl(id, camera, timestep?)` | **URL builder only** (for `<img src>`). **CRITICAL**: when `timestep` is defined, append **`?timestep=${timestep}`** to the camera URL. When omitted, document that behavior is for backward compatibility only; episode UIs should always pass timestep (see Pitfall #11 in parent README). |
| `getTokenMeta(id, timestep)` | `GET /api/episodes/{id}/timesteps/{timestep}/token-meta` → `TokenMeta[]` |
| `getAttention(id, timestep, layer, head, action)` | `GET .../attention?layer=&head=&action=` → `AttentionResponse` |
| `getAttentionSummary(id, timestep, layer, head)` | `GET .../attention/summary?layer=&head=` → `AttentionSummary` |
| `getTsne(id, timestep, layer)` | `GET .../tsne?layer=` → `TsneResponse` |
| `getNeighbors(id, timestep, layer, action)` | `GET .../tsne/neighbors?layer=&action=` → `NeighborResponse` |

Use **`SampledLayer`** for `layer` parameters and narrow `head` / `action` with constants from `constants.ts` where helpful.

### 5. MSW mock setup

- Add **request handlers** for all endpoints above returning **minimal but valid** JSON fixtures (867-length arrays where required can be shortened only in tests if documented; for dev browser mocks prefer correct lengths or a generator).
- Wire **MSW** in development (`main.tsx` or `src/mocks/browser.ts`) so **`npm run dev`** works with **no backend**.
- Expose a single flag or comment block explaining how to **disable MSW** when pointing at a live server (A5 formalizes this).

### 6. Routing (`src/App.tsx`)

- **`/`** → `HomePage` (placeholder that eventually lists episodes).
- **`/episode/:episodeId`** → `EpisodePage` (placeholder shell).

Use **`createBrowserRouter`** or `<Routes>` / `<Route>` per team preference; keep 404 behavior defined (simple “Unknown route” is fine).

## Acceptance Criteria

1. `npm run build` succeeds with **TypeScript strict** and no `any` in `client.ts` / `api.ts`.
2. All **nine** `api` methods exist and construct URLs matching the backend README (including query params).
3. **`getCameraImageUrl(id, camera, timestep)`** accepts optional `timestep` and, when provided, produces a URL containing **`?timestep=<N>`**.
4. **`SAMPLED_LAYERS`** lives in `api.ts`; **`constants.ts`** imports it (no duplicate layer array).
5. With MSW enabled, opening `/` and `/episode/demo` does not throw network errors for mocked calls you trigger from placeholder `useEffect` smoke calls (optional but recommended).
6. Router navigates between home and episode routes without full page reload.

## Out of Scope

- Real layouts, charts, and context (A2+).
- Playwright / E2E (optional backlog).
