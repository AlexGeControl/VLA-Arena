"""HDF5 serialization for extracted Pi-Zero episode data.

Writes one ``.h5`` file per episode matching the schema consumed by the
backend ``HDF5Reader``.

Schema
------
::

    {episode_id}.h5
      /meta  (attrs: episode_id, task_instruction, num_timesteps,
              instruction_tokens JSON, sampled_layers JSON)
      /cameras/{name}  (uint8 [H, W, 3])
      /timestep_NNN/token_meta  (JSON string)
      /timestep_NNN/attention/layer_XX  (float32 [8, 51, 867])
      /timestep_NNN/tsne/layer_XX  (float32 [867, 2])
      /timestep_NNN/neighbors/layer_XX  (compound [50, 5])
      /timestep_NNN/q_projections/layer_XX/{prefix,suffix}  (float32)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

SAMPLED_LAYERS: list[int] = [0, 3, 6, 9, 12, 15, 17]
NEIGHBOR_DTYPE = np.dtype([("neighbor_index", "<i4"), ("distance", "<f4")])


def write_episode_hdf5(
    output_dir: Path,
    episode_id: str,
    task_instruction: str,
    instruction_tokens: list[str],
    timestep_data: list[dict],
) -> Path:
    """Write a complete episode to an HDF5 file.

    Args:
        output_dir: Directory to write the file into.
        episode_id: Unique episode identifier (used as filename stem).
        task_instruction: Natural language task instruction.
        instruction_tokens: Tokenized instruction as list of strings.
        timestep_data: List of dicts, each containing per-timestep data.
            Required keys per dict: ``timestep``, ``token_meta``,
            ``attention``, ``tsne``, ``neighbors``, ``camera_images``.
            Optional: ``q_prefix``, ``q_suffix``.

    Returns:
        Path to the written HDF5 file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{episode_id}.h5"

    with h5py.File(path, "w") as f:
        _write_meta(f, episode_id, task_instruction, instruction_tokens, len(timestep_data))
        for ts_data in timestep_data:
            _write_timestep(f, ts_data)

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.1f MB, %d timesteps)", path.name, size_mb, len(timestep_data))
    return path


def _write_meta(
    f: h5py.File,
    episode_id: str,
    task_instruction: str,
    instruction_tokens: list[str],
    num_timesteps: int,
) -> None:
    """Write ``/meta`` group with episode-level attributes."""
    meta = f.create_group("meta")
    meta.attrs["episode_id"] = episode_id
    meta.attrs["task_instruction"] = task_instruction
    meta.attrs["num_timesteps"] = num_timesteps
    meta.attrs["instruction_tokens"] = json.dumps(instruction_tokens)
    meta.attrs["sampled_layers"] = json.dumps(SAMPLED_LAYERS)


def _write_cameras(f: h5py.File, camera_images: dict[str, np.ndarray]) -> None:
    """Write ``/cameras/{name}`` datasets."""
    if not camera_images:
        return
    cam_grp = f.create_group("cameras")
    for name, img in camera_images.items():
        cam_grp.create_dataset(name, data=img.astype(np.uint8), compression="gzip")


def _write_timestep(f: h5py.File, ts_data: dict) -> None:
    """Write all data for a single timestep, including camera images."""
    ts_idx = ts_data["timestep"]
    ts_key = f"timestep_{ts_idx:03d}"
    ts_grp = f.create_group(ts_key)

    _write_token_meta(ts_grp, ts_data["token_meta"])
    _write_cameras(ts_grp, ts_data.get("camera_images", {}))
    _write_attention(ts_grp, ts_data["attention"])
    _write_tsne(ts_grp, ts_data["tsne"])
    _write_neighbors(ts_grp, ts_data["neighbors"])
    _write_q_projections(ts_grp, ts_data.get("q_prefix", {}), ts_data.get("q_suffix", {}))
    _write_cmf_attended(ts_grp, ts_data.get("cmf_attended", {}))


def _write_token_meta(ts_grp: h5py.Group, token_meta: list[dict]) -> None:
    """Write ``token_meta`` as a JSON-encoded string dataset."""
    ts_grp.create_dataset("token_meta", data=json.dumps(token_meta))


def _write_attention(ts_grp: h5py.Group, attention: dict[int, np.ndarray]) -> None:
    """Write ``attention/layer_XX`` datasets (float32 [8, 51, 867])."""
    if not attention:
        return
    attn_grp = ts_grp.create_group("attention")
    for layer_idx, arr in attention.items():
        attn_grp.create_dataset(
            f"layer_{layer_idx:02d}",
            data=arr.astype(np.float32),
            chunks=True,
            compression="gzip",
        )


def _write_tsne(ts_grp: h5py.Group, tsne: dict[int, np.ndarray]) -> None:
    """Write ``tsne/layer_XX`` datasets (float32 [867, 2])."""
    if not tsne:
        return
    tsne_grp = ts_grp.create_group("tsne")
    for layer_idx, arr in tsne.items():
        tsne_grp.create_dataset(
            f"layer_{layer_idx:02d}",
            data=arr.astype(np.float32),
        )


def _write_neighbors(ts_grp: h5py.Group, neighbors: dict[int, np.ndarray]) -> None:
    """Write ``neighbors/layer_XX`` datasets (compound [50, 5])."""
    if not neighbors:
        return
    nbr_grp = ts_grp.create_group("neighbors")
    for layer_idx, arr in neighbors.items():
        nbr_grp.create_dataset(f"layer_{layer_idx:02d}", data=arr)


def _write_cmf_attended(
    ts_grp: h5py.Group,
    cmf_attended: dict[int, dict[str, np.ndarray]],
) -> None:
    """Write ``cmf_attended/layer_XX/{language,visual}`` datasets.

    Each value is float32 with shape ``(51, 8, 256)``.
    """
    if not cmf_attended:
        return
    cmf_grp = ts_grp.create_group("cmf_attended")
    for layer_idx in sorted(cmf_attended.keys()):
        layer_grp = cmf_grp.create_group(f"layer_{layer_idx:02d}")
        for modality, arr in cmf_attended[layer_idx].items():
            layer_grp.create_dataset(
                modality,
                data=arr.astype(np.float32),
                compression="gzip",
            )


def _write_q_projections(
    ts_grp: h5py.Group,
    q_prefix: dict[int, np.ndarray],
    q_suffix: dict[int, np.ndarray],
) -> None:
    """Write ``q_projections/layer_XX/{prefix,suffix}`` datasets."""
    if not q_prefix and not q_suffix:
        return
    qp_grp = ts_grp.create_group("q_projections")
    all_layers = set(q_prefix.keys()) | set(q_suffix.keys())
    for layer_idx in sorted(all_layers):
        layer_grp = qp_grp.create_group(f"layer_{layer_idx:02d}")
        if layer_idx in q_prefix:
            layer_grp.create_dataset(
                "prefix",
                data=q_prefix[layer_idx].astype(np.float32),
                compression="gzip",
            )
        if layer_idx in q_suffix:
            layer_grp.create_dataset(
                "suffix",
                data=q_suffix[layer_idx].astype(np.float32),
                compression="gzip",
            )
