"""Shared test fixtures for analytics tests.

Generates a minimal HDF5 file with attention, Q-projections, t-SNE,
and cmf_attended data for 1 episode / 2 timesteps / 7 sampled layers.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

SAMPLED_LAYERS = [0, 3, 6, 9, 12, 15, 17]
EPISODE_ID = "test_analytics_ep"
NUM_TIMESTEPS = 2


def _softmax_rows(arr: np.ndarray) -> np.ndarray:
    exp = np.exp(arr - arr.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def create_analytics_h5(path: Path, rng: np.random.Generator) -> None:
    """Write a test HDF5 with all fields needed by the analytics pipeline."""
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["episode_id"] = EPISODE_ID
        meta.attrs["task_instruction"] = "test task"
        meta.attrs["num_timesteps"] = NUM_TIMESTEPS
        meta.attrs["instruction_tokens"] = json.dumps(["test", "task"])
        meta.attrs["sampled_layers"] = json.dumps(SAMPLED_LAYERS)

        for t in range(NUM_TIMESTEPS):
            ts = f.create_group(f"timestep_{t:03d}")

            attn_grp = ts.create_group("attention")
            tsne_grp = ts.create_group("tsne")
            qproj_grp = ts.create_group("q_projections")
            cmf_grp = ts.create_group("cmf_attended")

            for layer in SAMPLED_LAYERS:
                lk = f"layer_{layer:02d}"

                raw_attn = rng.standard_normal((8, 51, 867)).astype(np.float32)
                attn_grp.create_dataset(lk, data=_softmax_rows(raw_attn))

                tsne_grp.create_dataset(
                    lk, data=rng.standard_normal((867, 2)).astype(np.float32)
                )

                lyr_q = qproj_grp.create_group(lk)
                lyr_q.create_dataset(
                    "prefix",
                    data=rng.standard_normal((816, 8, 256)).astype(np.float32),
                )
                lyr_q.create_dataset(
                    "suffix",
                    data=rng.standard_normal((51, 8, 256)).astype(np.float32),
                )

                lyr_cmf = cmf_grp.create_group(lk)
                lyr_cmf.create_dataset(
                    "language",
                    data=rng.standard_normal((51, 8, 256)).astype(np.float32),
                )
                lyr_cmf.create_dataset(
                    "visual",
                    data=rng.standard_normal((51, 8, 256)).astype(np.float32),
                )


@pytest.fixture()
def analytics_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with one analytics-ready HDF5 file."""
    rng = np.random.default_rng(42)
    h5_path = tmp_path / f"{EPISODE_ID}.h5"
    create_analytics_h5(h5_path, rng)
    return tmp_path


@pytest.fixture()
def analytics_reader(analytics_data_dir: Path):
    from analytics.reader import AnalyticsReader

    return AnalyticsReader(analytics_data_dir / f"{EPISODE_ID}.h5")
