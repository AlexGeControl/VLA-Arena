# Task A5: Polish, Brushing & Production Wiring

> Part of [Frontend Development (Track A)](../README.md). **Depends on [A3](../A3-attention-view/prompt.md) and [A4](../A4-embedding-view/prompt.md).**

## Goal

Complete **cross-panel brushing**, rich **popups** for language and image patches, shared **UI primitives**, **visual polish**, and **real backend integration** by disabling MSW and pointing the client at **`VITE_API_BASE`**.

## Task

### 1. Cross-panel brushing

Use **`highlightedTokenIndex`** in **`EpisodeContext`** as the single source of truth:

- **Attention → t-SNE**: hovering a language span or image patch (A3) sets **`highlightedTokenIndex`**. The **t-SNE** plot should **emphasize** that point (e.g. larger radius, pulse, or second ring) while hovered.
- **t-SNE → Attention**: hovering a scatter point sets **`highlightedTokenIndex`**. **LanguageAttention** and **ImageAttention** should show the same highlight state (A3 red rectangle + language emphasis).
- Clear highlight on mouse leave **unless** you intentionally implement “sticky selection” — default spec: **clear on leave** for both panels.

Ensure indices are **consistent** with the backend’s **867** token ordering.

### 2. `LanguageTokenPopup`

- Modal or anchored popover showing the **full task instruction**.
- The **hovered token** (by character span or token index) is **highlighted** (background, underline, or bold) without breaking readability.
- Trigger from **t-SNE tooltip** and optionally from attention language row.

### 3. `ImagePatchPopup`

- **128×128** thumbnail centered or pinned near cursor.
- **Dark semi-transparent overlay** on the full camera frame (or cropped context) with a **red border** outlining the **patch** region corresponding to `patch_row` / `patch_col`.
- **Pitfall #11**: component **must** accept **`timestep: number`** and call **`api.getCameraImageUrl(episodeId, camera, timestep)`**. Never omit timestep when showing a live frame from the episode.

### 4. Common components

- **`Badge`**: small pill for counts, layer/head labels, connection status.
- **`Skeleton`**: rectangular shimmer placeholders for attention bars, scatter loading, images.

### 5. Visual polish — CMU Brand Alignment

The visual theme follows [CMU Brand Standards](https://brand.cmu.edu/visual-identity/colors).

**UI Chrome (Core Colors)**:
- **Header bars**: Carnegie Red `#C41230` background with white text. Carnegie Red must dominate attention as the brand hero color.
- **Page background**: White `#FFFFFF`.
- **Card/panel borders**: Steel Gray `#E0E0E0`.
- **Loading spinners**: Carnegie Red border-top on Steel Gray border.
- **Attention heatmap**: Carnegie Red ramp `rgba(196,18,48,alpha)` — NOT Tailwind red-500.

**Data Visualization Colors** (defined in `TOKEN_COLORS`, Tailwind config `colors.cmu.*`):
- Base camera: Blue Thread `#043673`, Left wrist: Sky Blue `#007BC0`, Right wrist: Hornbostel Teal `#1F4C4C`
- Language: Gold Thread `#FDB515`, State: Green Thread `#009647`, Action: Carnegie Red `#C41230`
- The summary bar language segment uses dark text (Gold Thread is light).

**t-SNE Scatter Plot** (Pitfall #13):
- SVG must have **white background** (`backgroundColor: '#FFFFFF'`) with Steel Gray border.
- Selected/highlighted token strokes must be **black** (`#000000`), NOT white — the CMU token colors are dark.
- Neighbor highlight rings: black, 2px. Neighbor dashed lines: Iron Gray at 60% opacity.

**Fonts**: **Inter** for UI body, **JetBrains Mono** for numeric readouts. Load via Google Fonts.
**Loading states**: every async view shows **Skeleton** or spinner; no blank panels.
**Responsive**: TimestepSlider and SharedControls wrap gracefully; scatter usable at tablet width.

### 6. Real data integration

- **Disable MSW** in production / “live” mode:
  - Document env flag (e.g. `VITE_USE_MSW=false`) or remove `worker.start()` from `main.tsx` when `import.meta.env.PROD` or when **`VITE_API_BASE`** points to a real host.
- Ensure **CORS** is satisfied (backend allows SPA origin or `*` per v3 design).
- **`.env.example`**: `VITE_API_BASE=http://localhost:8080/api`, `VITE_USE_MSW=true` for local mock dev.

## Acceptance Criteria

1. Hovering in **Attention** updates **t-SNE** highlight and vice versa via **`highlightedTokenIndex`** only (no duplicate hover state).
2. **`LanguageTokenPopup`** shows full instruction with **clear** hovered-token emphasis.
3. **`ImagePatchPopup`** renders correct patch for the **current timestep** (visual check: scrub timestep — patch background image updates).
4. **`Badge`** and **`Skeleton`** are reused in at least two places each.
5. With MSW off and backend running, episode page loads **real** JSON/PNG without console errors.
6. Typography matches spec (**Inter** + **JetBrains Mono**); header uses **Carnegie Red** `#C41230`.
7. t-SNE scatter has **white background**, **black strokes** on highlights, all 6 CMU token colors visible.
8. Attention heatmap uses Carnegie Red ramp (`rgba(196,18,48,alpha)`), NOT Tailwind red.
9. Summary bar language segment uses **dark text** on Gold Thread background.

## Out of Scope

- Playwright E2E, authentication, deployment pipelines (backlog).
