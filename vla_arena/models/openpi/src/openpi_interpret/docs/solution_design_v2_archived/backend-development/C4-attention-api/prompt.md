# Task C4: Attention API

> Part of the [Backend Development](../README.md) epic. Depends on [C2](../C2-hdf5-data-layer/prompt.md).

## Goal

REST endpoints for serving attention weights — both the raw attention row for a selected action token and a pre-computed modality breakdown.

## Task

### Endpoints

Create `app/routers/attention.py`:

**`GET /api/episodes/{id}/timesteps/{t}/attention`** — Attention row + breakdown for a selected head and action token.

Query parameters:
- `layer` (int, required): One of sampled layers `[0, 3, 6, 9, 12, 15, 17]`
- `head` (int, required): `0–7`
- `action` (int, required): `0–49`

```python
@router.get("/api/episodes/{episode_id}/timesteps/{timestep}/attention",
            response_model=AttentionResponse)
async def get_attention(
    episode_id: str, timestep: int,
    layer: int, head: int, action: int, ...
):
    attn = reader.get_attention(timestep, layer)  # float32 [8, 51, 867]
    suffix_query = 1 + action  # +1 for state token at position 0
    row = attn[head, suffix_query, :]  # float32 [867]
    breakdown = compute_breakdown(row)
    return AttentionResponse(row=row.tolist(), breakdown=breakdown)
```

### Breakdown Computation

The 867-element attention row is sliced by the token map:

```python
TOKEN_RANGES = {
    "base_0_rgb":        (0, 256),
    "left_wrist_0_rgb":  (256, 512),
    "right_wrist_0_rgb": (512, 768),
    "language":          (768, 816),
    "state":             (816, 817),
    "action":            (817, 867),
}

def compute_breakdown(row: np.ndarray) -> dict:
    cameras = {}
    camera_totals = {}
    for cam in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        start, end = TOKEN_RANGES[cam]
        weights = row[start:end]
        cameras[cam] = weights.tolist()
        camera_totals[cam] = float(weights.sum())

    lang_start, lang_end = TOKEN_RANGES["language"]
    language_weights = row[lang_start:lang_end].tolist()

    state_start, state_end = TOKEN_RANGES["state"]
    state_weight = float(row[state_start])

    act_start, act_end = TOKEN_RANGES["action"]
    action_weights = row[act_start:act_end].tolist()

    return {
        "cameras": cameras,
        "camera_totals": camera_totals,
        "language_weights": language_weights,
        "language_total": float(sum(language_weights)),
        "state_weight": state_weight,
        "action_weights": action_weights,
        "action_total": float(sum(action_weights)),
    }
```

**`GET /api/episodes/{id}/timesteps/{t}/attention/summary`** — Modality-level attention summary across all action tokens.

Query parameters:
- `layer` (int, required)
- `head` (int, required)

```python
@router.get("/api/episodes/{episode_id}/timesteps/{timestep}/attention/summary")
async def get_attention_summary(
    episode_id: str, timestep: int, layer: int, head: int, ...
):
    attn = reader.get_attention(timestep, layer)  # [8, 51, 867]
    action_rows = attn[head, 1:, :]  # [50, 867] — skip state token

    modality_totals = {
        "images": float(action_rows[:, 0:768].sum(axis=1).mean()),
        "language": float(action_rows[:, 768:816].sum(axis=1).mean()),
        "state": float(action_rows[:, 816].mean()),
        "actions": float(action_rows[:, 817:867].sum(axis=1).mean()),
    }

    per_action_to_prefix = action_rows[:, :816].sum(axis=1).tolist()

    return {"modality_totals": modality_totals, "per_action": per_action_to_prefix}
```

### Validation

- `layer` must be in `[0, 3, 6, 9, 12, 15, 17]` (422 otherwise)
- `head` must be in `[0, 7]` (422 otherwise)
- `action` must be in `[0, 49]` (422 otherwise)

## Acceptance Criteria

- [ ] `GET .../attention?layer=0&head=0&action=0` returns a 867-element row and breakdown
- [ ] Breakdown camera arrays have 256 elements each
- [ ] Breakdown totals sum to ~1.0 (±float rounding)
- [ ] `GET .../attention/summary` returns modality_totals and per_action arrays
- [ ] Invalid layer/head/action returns 422
- [ ] Response time < 50ms for a single attention query (numpy slicing is instant)
