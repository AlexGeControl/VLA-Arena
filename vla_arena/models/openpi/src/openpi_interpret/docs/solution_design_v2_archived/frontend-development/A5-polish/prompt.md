# Task A5: Polish, Cross-Panel Brushing & Contextual Pop-ups

## Context

Tasks A1–A4 have delivered a working application with episode navigation, shared controls, attention visualization, and t-SNE embedding visualization. This final task adds **cross-panel brushing** (bidirectional highlighting between the two panels), contextual pop-ups for the t-SNE view, and visual polish across the entire application.

**Key change from v1**: The panels now share controls (layer, head, action token) via `EpisodeContext`. This phase adds the final piece of cross-panel interaction: **`highlightedTokenIndex`** — when the user hovers over a token in one panel, the corresponding element in the other panel is highlighted.

**Prerequisite**: Tasks A3 and A4 are both complete and working.

## Task

### 1. Cross-panel brushing

Implement bidirectional highlighting between the Attention View and Embedding View using the `highlightedTokenIndex` field in `EpisodeContext`.

**Attention View → Embedding View**:
- When the user hovers over a **language token span** in the Attention View, set `highlightedTokenIndex` to that token's index (768–815).
- When the user hovers over an **image patch** in the Attention View (on the canvas overlay), calculate the patch's token index from its (row, col) and camera, and set `highlightedTokenIndex`.
- The Embedding View listens for `highlightedTokenIndex` changes and:
  - Enlarges the corresponding point in the scatter plot (radius 5px, white stroke)
  - Optionally smoothly scrolls the SVG to center on the highlighted point if it's outside the viewport.
- On mouse leave, set `highlightedTokenIndex = null` to clear the highlight.

**Embedding View → Attention View**:
- When the user hovers over a **language token point** in the t-SNE scatter, set `highlightedTokenIndex`.
- The Attention View listens and applies a **pulse/glow** CSS animation to the corresponding language token span.
- When the user hovers over an **image patch point** in the t-SNE scatter, set `highlightedTokenIndex`.
- The Attention View listens and draws a highlight rectangle on the corresponding patch position in the camera canvas overlay.
- On mouse leave, clear the highlight.

**Implementation**:
- Use `EpisodeContext.highlightedTokenIndex` (already declared in Task A2's context interface).
- Both panels use a `useEffect` that watches `highlightedTokenIndex` and applies/removes visual effects.
- Keep the hover handler lightweight — avoid re-rendering the full scatter plot. Use `useRef` for the SVG element and mutate the DOM directly for the highlight circle.

### 2. Language token pop-up (t-SNE view)

When the user hovers over a **language token** in the t-SNE scatter plot, enhance the tooltip with a contextual pop-up that shows the full tokenized instruction with the hovered token highlighted:

- Display the pop-up below the basic tooltip (or as an expanded tooltip section).
- Render the full instruction as a horizontal sequence of subword token spans (same style as the Attention View's language display from Task A3).
- The hovered token's span should have a highlighted background (bright orange with white text) while all other spans have a neutral gray background.

Example for hovering over the token "shelf":
```
┌──────────────────────────────────────────────────┐
│  token: "shelf" (position 10)                     │
│                                                    │
│  pick  up  the  red  cup  and  place  it  on  the │
│  ░░░░  ░░  ░░░  ░░░  ░░░  ░░░  ░░░░░  ░░  ░░  ░░│
│  █████                                             │
│  shelf                                             │
└──────────────────────────────────────────────────┘
```

### 3. Image patch pop-up (t-SNE view)

When the user hovers over an **image patch token** in the t-SNE scatter plot, enhance the tooltip with a contextual pop-up showing the source camera image with the hovered patch highlighted:

- Display the source camera image at a thumbnail size (128×128 px).
- Overlay a semi-transparent dark mask over the entire image.
- Draw a red border rectangle at the patch's position.
- Pixel coordinates: `(col * 8, row * 8)` to `((col+1) * 8, (row+1) * 8)` since 128px / 16 patches = 8px per patch.
- Show the camera name and patch coordinates: e.g., "base_0_rgb patch (row 5, col 12)".

Implementation:
1. Render the camera image on a small `<canvas>` (128×128).
2. Draw a semi-transparent black overlay (`rgba(0,0,0,0.5)`) over the whole canvas.
3. Clear the overlay at the patch's region.
4. Draw a 2px red border rectangle around the patch region.

### 4. Visual polish

Apply the following refinements across the entire application:

**Typography & spacing**:
- Use a consistent font stack: `Inter` for UI text (import from Google Fonts), `JetBrains Mono` for numeric values (attention percentages, coordinates).
- Consistent heading hierarchy: panel titles as `text-lg font-semibold`, section headers as `text-sm font-medium uppercase tracking-wide text-gray-500`.
- Adequate spacing between sections (at least `gap-4` or `space-y-4`).

**Color & theme**:
- Dark header bar with white text for the app title.
- White/light-gray panel backgrounds with subtle borders (`border border-gray-200 rounded-lg`).
- Consistent use of the 6 modality colors from `TOKEN_COLORS` across both panels.

**Loading & empty states**:
- Show a skeleton loader (gray pulsing rectangles) while episode data or attention data is being fetched.
- Show a friendly empty state if no episodes are available.
- Show "No data for this selection" if a timestep has no attention/embedding data.

**Responsive layout**:
- On screens narrower than 1024px, the two panels should stack vertically (Attention View on top, Embedding View below).
- Camera images in the Attention View should wrap to 2 columns or 1 column on narrow screens.
- The t-SNE scatter plot should scale down proportionally.

**Accessibility basics**:
- All dropdowns should have associated `<label>` elements.
- Color-coded elements should have text labels as well (don't rely on color alone).
- Focus-visible outlines on interactive elements.

### 5. Real data integration

If Tracks B and C have completed by this point:
- Disable MSW mocks and point the frontend at the live FastAPI backend (`VITE_API_BASE=http://localhost:8080/api`).
- Verify all visualizations work with real attention patterns and t-SNE coordinates from the backend.
- Adjust any hardcoded assumptions (e.g., number of actual language tokens may be < 48).

If Tracks B/C are not yet complete, this step is deferred — the app continues to work with MSW mock responses.

### Component structure changes

```
src/components/
  common/
    Skeleton.tsx               # Reusable loading skeleton
    Badge.tsx                  # Percentage badge component
  embedding/
    LanguageTokenPopup.tsx     # Contextual instruction pop-up
    ImagePatchPopup.tsx        # Contextual image patch pop-up
```

### Acceptance Criteria

- [ ] Hovering a language span in the Attention View highlights the corresponding point in the t-SNE scatter
- [ ] Hovering a language point in the t-SNE scatter applies a glow to the corresponding span in the Attention View
- [ ] Hovering an image patch in either panel highlights it in the other panel
- [ ] Highlights clear immediately on mouse leave (no stale highlights)
- [ ] Hovering a language token in the t-SNE view shows the full instruction with that token highlighted in the pop-up
- [ ] Hovering an image patch token in the t-SNE view shows the camera thumbnail with the patch location marked
- [ ] Pop-ups dismiss cleanly when the cursor leaves the point
- [ ] Consistent typography, spacing, and color usage across the entire app
- [ ] Loading skeletons appear while data is being fetched
- [ ] Layout responds correctly at 1440px, 1024px, and 768px viewport widths
- [ ] All dropdowns have accessible labels
- [ ] No TypeScript errors, no console warnings, no layout shifts during interaction
