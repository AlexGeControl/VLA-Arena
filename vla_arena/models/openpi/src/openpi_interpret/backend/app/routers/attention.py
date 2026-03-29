"""Attention row, breakdown, and summary endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.data.constants import SAMPLED_LAYERS, TOKEN_RANGES
from app.data.dependencies import get_episode_index
from app.data.hdf5_reader import EpisodeIndex
from app.data.schemas import (
    AttentionBreakdownDetail,
    AttentionResponse,
    AttentionSummary,
)

router = APIRouter(
    prefix="/api/episodes/{episode_id}/timesteps/{timestep}/attention",
    tags=["attention"],
)

NUM_HEADS = 8
NUM_ACTIONS = 50
CAMERA_KEYS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]


def _validate_params(
    layer: int, head: int, action: int | None = None
) -> None:
    """Raise HTTPException(422) for out-of-range attention parameters."""
    if layer not in SAMPLED_LAYERS:
        raise HTTPException(
            status_code=422, detail=f"Invalid layer {layer}"
        )
    if not 0 <= head < NUM_HEADS:
        raise HTTPException(
            status_code=422, detail=f"Invalid head {head}"
        )
    if action is not None and not 0 <= action < NUM_ACTIONS:
        raise HTTPException(
            status_code=422, detail=f"Invalid action {action}"
        )


def _get_reader(episode_id: str, index: EpisodeIndex):
    """Return an HDF5Reader or raise 404."""
    try:
        return index.get_reader(episode_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Episode not found")


@router.get("", response_model=AttentionResponse)
async def get_attention(
    episode_id: str,
    timestep: int,
    layer: int = Query(...),
    head: int = Query(...),
    action: int = Query(...),
    index: EpisodeIndex = Depends(get_episode_index),
) -> AttentionResponse:
    """Return one attention row and its modality breakdown.

    The suffix query index is ``1 + action`` because the state token
    sits at suffix position 0.
    """
    _validate_params(layer, head, action)
    reader = _get_reader(episode_id, index)
    attn = reader.get_attention(timestep, layer)
    suffix_query = 1 + action
    row = attn[head, suffix_query, :].tolist()
    breakdown = _compute_breakdown(row)
    return AttentionResponse(row=row, breakdown=breakdown)


@router.get("/summary", response_model=AttentionSummary)
async def get_attention_summary(
    episode_id: str,
    timestep: int,
    layer: int = Query(...),
    head: int = Query(...),
    index: EpisodeIndex = Depends(get_episode_index),
) -> AttentionSummary:
    """Aggregate attention across all 50 action queries."""
    _validate_params(layer, head)
    reader = _get_reader(episode_id, index)
    attn = reader.get_attention(timestep, layer)

    modality_totals: dict[str, float] = {}
    per_action: list[float] = []

    for action_idx in range(NUM_ACTIONS):
        suffix_query = 1 + action_idx
        row = attn[head, suffix_query, :]
        per_action.append(float(row.sum()))
        for key, (start, end) in TOKEN_RANGES.items():
            segment_sum = float(row[start:end].sum())
            modality_totals[key] = modality_totals.get(key, 0.0) + segment_sum

    total = sum(modality_totals.values()) or 1.0
    modality_totals = {k: v / total for k, v in modality_totals.items()}
    return AttentionSummary(
        modality_totals=modality_totals, per_action=per_action
    )


def _compute_breakdown(row: list[float]) -> AttentionBreakdownDetail:
    """Slice the 867-element attention row into modality segments."""
    cameras: dict[str, list[float]] = {}
    camera_totals: dict[str, float] = {}

    for cam_key in CAMERA_KEYS:
        start, end = TOKEN_RANGES[cam_key]
        weights = row[start:end]
        cameras[cam_key] = weights
        camera_totals[cam_key] = sum(weights)

    lang_start, lang_end = TOKEN_RANGES["language"]
    language_weights = row[lang_start:lang_end]

    state_start, state_end = TOKEN_RANGES["state"]
    state_weight = sum(row[state_start:state_end])

    action_start, action_end = TOKEN_RANGES["action"]
    action_weights = row[action_start:action_end]

    return AttentionBreakdownDetail(
        cameras=cameras,
        camera_totals=camera_totals,
        language_weights=language_weights,
        language_total=sum(language_weights),
        state_weight=state_weight,
        action_weights=action_weights,
        action_total=sum(action_weights),
    )
