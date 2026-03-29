# C4 — Attention API

## Goal

Expose suffix-only cross-attention: one **867-dimensional** key row per (layer, head, action query), plus aggregated **summary** over all action queries. All math uses pre-stored weights only (no ML).

## Router

- Prefix: **`/api/episodes/{episode_id}/timesteps/{timestep}/attention`**
- Tag: e.g. `attention`

## Suffix Query Index (Critical)

Attention tensor shape: **`(8, 51, 867)`** — 8 heads, **51** suffix queries, 867 keys.

- Suffix position **0** = **state** token.
- Action tokens are suffix positions **1..50**.
- Query parameter **`action`** is **`0..49`** (action index).
- **Row selection:** `suffix_query = 1 + action`, then `row = attn[head, suffix_query, :]` (867 floats).

Document this in the route docstring so frontend authors do not off-by-one.

## `GET ""` — full row + breakdown

**Query params (required):** `layer`, `head`, `action` (all `int`).

**Response:** `AttentionResponse` — `row: list[float]` (length 867), `breakdown: AttentionBreakdownDetail`.

**Validation (422 if invalid):**

- `layer in SAMPLED_LAYERS` (`[0,3,6,9,12,15,17]`)
- `0 <= head < 8`
- `0 <= action < 50`

**Implementation:**

1. Load `attn = reader.get_attention(timestep, layer)`.
2. Extract row as above.
3. **`_compute_breakdown(row)`** — slice using **`TOKEN_RANGES`**:
   - For each camera key in `["base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb"]`: `row[start:end]` → list in `cameras[cam_key]`, sum → `camera_totals[cam_key]`.
   - `language_weights = row[768:816]`, `language_total = sum(...)`.
   - `state_weight = sum(row[816:817])`.
   - `action_weights = row[817:867]`, `action_total = sum(...)`.

**404** if episode not found; **422** for bad layer/head/action; **404/500** if dataset missing (choose consistent policy—typically 404 for missing layer dataset).

## `GET "/summary"` — aggregate over actions

**Query params:** `layer`, `head` (required).

**Response:** `AttentionSummary`:

- `per_action`: **50** floats — for each `action_idx`, `suffix_query = 1 + action_idx`, `per_action[i] = float(attn[head, suffix_query, :].sum())` (total mass per action query).
- `modality_totals`: dict keyed like `TOKEN_RANGES` — for each action row, add segment sums per modality, then **normalize** so values sum to 1.0 across modalities (divide by total sum of all modality sums, guard zero).

**Validation:** same layer/head rules as above (no action param).

## Files

- `app/routers/attention.py` — router + `_compute_breakdown`, `_validate_params`, small helpers.
- Include router in `main.py`.

## Acceptance Criteria

1. For valid fixture data, attention row length is **867**.
2. `action=0` uses suffix row index **1**; `action=49` uses index **50** (unit test with distinguishable rows if fixture allows, or tensor filled with sentinel values in test HDF5).
3. `breakdown` sums (`language_total + state_weight + action_total + sum(camera_totals)`) approximately equal `sum(row)` (within float tolerance).
4. `summary.per_action` has length **50**; `modality_totals` has keys for all six `TOKEN_RANGES` entries and floats sum to ~1.0 after normalization.
5. Invalid `layer` / `head` / `action` → **422**.
6. Tests in `tests/test_attention.py` cover row shape, validation, and breakdown consistency.
