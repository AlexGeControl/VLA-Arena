"""Shared test fixtures: HDF5 file generator and FastAPI TestClient."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.data.constants import (
    CAMERA_NAMES,
    MODALITY_GROUPS,
    NEIGHBOR_DTYPE,
    SAMPLED_LAYERS,
    TOKEN_RANGES,
)
from app.data.hdf5_reader import EpisodeIndex

EPISODE_ID = "test_ep_001"
TASK_INSTRUCTION = "pick up the red cup"
INSTRUCTION_TOKENS = ["pick", "up", "the", "red", "cup"]
NUM_TIMESTEPS = 2
TEST_CAMERAS = ["base_0_rgb", "left_wrist_0_rgb"]


def _build_token_meta() -> list[dict[str, object]]:
    """Generate 867 token metadata entries matching the production layout."""
    tokens: list[dict[str, object]] = []
    for cam in CAMERA_NAMES:
        start, end = TOKEN_RANGES[cam]
        grid_size = 16
        for i in range(start, end):
            local = i - start
            tokens.append(
                {
                    "index": i,
                    "type": "image_patch",
                    "source": cam,
                    "patch_row": local // grid_size,
                    "patch_col": local % grid_size,
                }
            )

    lang_start, lang_end = TOKEN_RANGES["language"]
    for i in range(lang_start, lang_end):
        pos = i - lang_start
        tok_text = (
            INSTRUCTION_TOKENS[pos] if pos < len(INSTRUCTION_TOKENS) else ""
        )
        tokens.append(
            {
                "index": i,
                "type": "language",
                "source": "language",
                "token_text": tok_text,
                "token_position": pos,
            }
        )

    s_start, _ = TOKEN_RANGES["state"]
    tokens.append(
        {"index": s_start, "type": "state", "source": "state"}
    )

    a_start, a_end = TOKEN_RANGES["action"]
    for i in range(a_start, a_end):
        tokens.append(
            {
                "index": i,
                "type": "action",
                "source": "action",
            }
        )

    return tokens


def _softmax_rows(arr: np.ndarray) -> np.ndarray:
    """Apply softmax along the last axis."""
    exp = np.exp(arr - arr.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _build_neighbors(rng: np.random.Generator) -> np.ndarray:
    """Build structured neighbor array [50, 5]."""
    dt = np.dtype(NEIGHBOR_DTYPE)
    data = np.empty((50, 5), dtype=dt)
    for action_idx in range(50):
        for mod_idx in range(5):
            start, end = list(TOKEN_RANGES.values())[mod_idx]
            data[action_idx, mod_idx]["neighbor_index"] = rng.integers(
                start, end
            )
            data[action_idx, mod_idx]["distance"] = rng.uniform(0.1, 5.0)
    return data


def create_test_h5(h5_path: Path) -> None:
    """Write a complete test HDF5 file to *h5_path*.

    Creates 1 episode with 2 timesteps, 2 cameras, attention, t-SNE,
    neighbors, and Q-projection data at all 7 sampled layers.
    """
    rng = np.random.default_rng(42)
    token_meta_json = json.dumps(_build_token_meta())

    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["episode_id"] = EPISODE_ID
        meta.attrs["task_instruction"] = TASK_INSTRUCTION
        meta.attrs["num_timesteps"] = NUM_TIMESTEPS
        meta.attrs["instruction_tokens"] = json.dumps(INSTRUCTION_TOKENS)
        meta.attrs["sampled_layers"] = json.dumps(SAMPLED_LAYERS)

        cams = f.create_group("cameras")
        for cam_name in TEST_CAMERAS:
            cams.create_dataset(
                cam_name,
                data=rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            )

        for t in range(NUM_TIMESTEPS):
            ts = f.create_group(f"timestep_{t:03d}")
            ts.create_dataset("token_meta", data=token_meta_json)

            attn_grp = ts.create_group("attention")
            tsne_grp = ts.create_group("tsne")
            nbr_grp = ts.create_group("neighbors")
            qproj_grp = ts.create_group("q_projections")

            for layer in SAMPLED_LAYERS:
                layer_key = f"layer_{layer:02d}"

                raw_attn = rng.standard_normal((8, 51, 867)).astype(
                    np.float32
                )
                attn_grp.create_dataset(
                    layer_key, data=_softmax_rows(raw_attn)
                )

                tsne_grp.create_dataset(
                    layer_key,
                    data=rng.standard_normal((867, 2)).astype(np.float32),
                )

                nbr_grp.create_dataset(
                    layer_key, data=_build_neighbors(rng)
                )

                lyr_qproj = qproj_grp.create_group(layer_key)
                lyr_qproj.create_dataset(
                    "prefix",
                    data=rng.standard_normal((816, 8, 256)).astype(
                        np.float32
                    ),
                )
                lyr_qproj.create_dataset(
                    "suffix",
                    data=rng.standard_normal((51, 8, 256)).astype(
                        np.float32
                    ),
                )


@pytest.fixture()
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with one test HDF5 file."""
    h5_path = tmp_path / f"{EPISODE_ID}.h5"
    create_test_h5(h5_path)
    return tmp_path


@pytest.fixture()
def episode_index(test_data_dir: Path) -> EpisodeIndex:
    """Return an EpisodeIndex pointing to the test data directory."""
    return EpisodeIndex(test_data_dir)


@pytest.fixture()
def client(test_data_dir: Path) -> TestClient:
    """Return a FastAPI TestClient backed by the test HDF5 fixture."""
    from app.main import app

    app.state.episode_index = EpisodeIndex(test_data_dir)
    return TestClient(app)
