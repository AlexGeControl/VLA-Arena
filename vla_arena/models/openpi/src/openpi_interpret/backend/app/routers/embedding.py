"""t-SNE scatter and nearest-neighbor endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.data.constants import (
    MODALITY_GROUPS,
    SAMPLED_LAYERS,
    TOKEN_COLORS,
    TOKEN_RANGES,
)
from app.data.dependencies import get_episode_index
from app.data.hdf5_reader import EpisodeIndex
from app.data.schemas import (
    NearestNeighbor,
    NeighborResponse,
    SelectedPoint,
    TsnePoint,
    TsneResponse,
)

router = APIRouter(
    prefix="/api/episodes/{episode_id}/timesteps/{timestep}/tsne",
    tags=["embedding"],
)

NUM_ACTIONS = 50


def _validate_layer(layer: int) -> None:
    """Raise 422 for invalid layer."""
    if layer not in SAMPLED_LAYERS:
        raise HTTPException(
            status_code=422, detail=f"Invalid layer {layer}"
        )


def _get_reader(episode_id: str, index: EpisodeIndex):
    """Return an HDF5Reader or raise 404."""
    try:
        return index.get_reader(episode_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Episode not found")


def _color_for_meta(meta: dict[str, object]) -> str:
    """Determine the display colour for a token based on its metadata."""
    if meta["type"] == "image_patch":
        color_key = str(meta["source"])
    else:
        color_key = str(meta["type"])
    return TOKEN_COLORS.get(color_key, "#888888")


@router.get("", response_model=TsneResponse)
async def get_tsne(
    episode_id: str,
    timestep: int,
    layer: int = Query(...),
    index: EpisodeIndex = Depends(get_episode_index),
) -> TsneResponse:
    """Return 867 t-SNE points with token metadata and colours."""
    _validate_layer(layer)
    reader = _get_reader(episode_id, index)
    coords = reader.get_tsne(timestep, layer)
    token_metas = reader.get_token_meta(timestep)
    points = _build_points(coords, token_metas)
    return TsneResponse(points=points)


def _build_points(
    coords, token_metas: list[dict[str, object]]
) -> list[TsnePoint]:
    """Assemble TsnePoint list from coordinates and metadata."""
    points: list[TsnePoint] = []
    for i, meta in enumerate(token_metas):
        points.append(
            TsnePoint(
                index=i,
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                type=str(meta["type"]),
                source=str(meta["source"]),
                color=_color_for_meta(meta),
            )
        )
    return points


@router.get("/neighbors", response_model=NeighborResponse)
async def get_neighbors(
    episode_id: str,
    timestep: int,
    layer: int = Query(...),
    action: int = Query(...),
    index: EpisodeIndex = Depends(get_episode_index),
) -> NeighborResponse:
    """Return 5 nearest neighbours for the specified action token."""
    _validate_layer(layer)
    if not 0 <= action < NUM_ACTIONS:
        raise HTTPException(
            status_code=422, detail=f"Invalid action {action}"
        )
    reader = _get_reader(episode_id, index)
    coords = reader.get_tsne(timestep, layer)
    nbr_data = reader.get_neighbors(timestep, layer)
    token_metas = reader.get_token_meta(timestep)

    action_global = TOKEN_RANGES["action"][0] + action
    selected = SelectedPoint(
        index=action_global,
        x=float(coords[action_global, 0]),
        y=float(coords[action_global, 1]),
    )

    neighbors = _build_neighbors(
        nbr_data, action, coords, token_metas
    )
    return NeighborResponse(selected=selected, neighbors=neighbors)


def _build_neighbors(
    nbr_data,
    action: int,
    coords,
    token_metas: list[dict[str, object]],
) -> list[NearestNeighbor]:
    """Extract 5 neighbor records for the given action index."""
    neighbors: list[NearestNeighbor] = []
    for mod_idx, mod_name in enumerate(MODALITY_GROUPS):
        entry = nbr_data[action, mod_idx]
        n_idx = int(entry["neighbor_index"])
        dist = float(entry["distance"])
        meta = token_metas[n_idx]
        neighbors.append(
            NearestNeighbor(
                index=n_idx,
                x=float(coords[n_idx, 0]),
                y=float(coords[n_idx, 1]),
                distance=dist,
                modality_group=mod_name,
                type=str(meta["type"]),
                source=str(meta["source"]),
            )
        )
    return neighbors
