"""Episode listing, metadata, camera image, and token-meta endpoints."""

from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from PIL import Image

from app.data.dependencies import get_episode_index
from app.data.hdf5_reader import EpisodeIndex
from app.data.schemas import EpisodeMeta, EpisodeSummary, TokenMeta

router = APIRouter(prefix="/api/episodes", tags=["episodes"])


@router.get("", response_model=list[EpisodeSummary])
async def list_episodes(
    index: EpisodeIndex = Depends(get_episode_index),
) -> list[EpisodeSummary]:
    """Return a compact summary for every available episode."""
    summaries: list[EpisodeSummary] = []
    for ep_id in index.list_ids():
        reader = index.get_reader(ep_id)
        meta = reader.get_meta()
        summaries.append(
            EpisodeSummary(
                episode_id=meta["episode_id"],
                task_instruction=meta["task_instruction"],
                num_timesteps=meta["num_timesteps"],
            )
        )
    return summaries


@router.get("/{episode_id}", response_model=EpisodeMeta)
async def get_episode(
    episode_id: str,
    index: EpisodeIndex = Depends(get_episode_index),
) -> EpisodeMeta:
    """Return full metadata for a single episode."""
    try:
        reader = index.get_reader(episode_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Episode not found")
    meta = reader.get_meta()
    return EpisodeMeta(**meta)


@router.get("/{episode_id}/camera/{camera_name}")
async def get_camera_image(
    episode_id: str,
    camera_name: str,
    timestep: int | None = None,
    index: EpisodeIndex = Depends(get_episode_index),
) -> Response:
    """Return a camera image as PNG binary.

    Args:
        timestep: Optional sequential timestep index. When provided, returns
            the per-timestep image (camera view at that point in time).
            Falls back to episode-level image if per-timestep data is absent.
    """
    try:
        reader = index.get_reader(episode_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Episode not found")
    try:
        img_array: np.ndarray = reader.get_camera_image(camera_name, timestep=timestep)
    except KeyError:
        raise HTTPException(status_code=404, detail="Camera not found")
    except ValueError:
        raise HTTPException(status_code=404, detail="Timestep not found")
    return _encode_png(img_array)


@router.get(
    "/{episode_id}/timesteps/{timestep}/token-meta",
    response_model=list[TokenMeta],
)
async def get_token_meta(
    episode_id: str,
    timestep: int,
    index: EpisodeIndex = Depends(get_episode_index),
) -> list[TokenMeta]:
    """Return per-token metadata for the given timestep."""
    try:
        reader = index.get_reader(episode_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Episode not found")
    try:
        raw_list = reader.get_token_meta(timestep)
    except ValueError:
        raise HTTPException(status_code=404, detail="Timestep not found")
    return [TokenMeta(**tok) for tok in raw_list]


def _encode_png(img_array: np.ndarray) -> Response:
    """Encode a uint8 numpy array to a PNG Response."""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
