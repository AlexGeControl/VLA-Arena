"""Thin h5py wrapper for analytics reads.

Provides typed access to attention, Q-projections, CMF attended
representations, and t-SNE coordinates from episode HDF5 files.
Independent of the backend's HDF5Reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from analytics.constants import SAMPLED_LAYERS


class AnalyticsReader:
    """Read-only accessor for a single episode HDF5 file.

    Args:
        path: Path to the .h5 file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._ts_keys: list[str] | None = None

    @property
    def episode_id(self) -> str:
        with h5py.File(self._path, "r") as f:
            return str(f["meta"].attrs["episode_id"])

    def get_timestep_keys(self) -> list[str]:
        """Return sorted timestep group names."""
        if self._ts_keys is None:
            with h5py.File(self._path, "r") as f:
                self._ts_keys = sorted(
                    k for k in f.keys() if k.startswith("timestep_")
                )
        return self._ts_keys

    @property
    def num_timesteps(self) -> int:
        return len(self.get_timestep_keys())

    def _resolve_ts_key(self, timestep: int) -> str:
        """Map sequential index to actual HDF5 group name."""
        keys = self.get_timestep_keys()
        if timestep < 0 or timestep >= len(keys):
            raise ValueError(
                f"Timestep {timestep} out of range [0, {len(keys)})"
            )
        return keys[timestep]

    def get_attention(self, timestep: int, layer: int) -> np.ndarray:
        """Read attention weights: float32 [8, 51, 867].

        Args:
            timestep: Sequential timestep index.
            layer: Sampled layer index.
        """
        self._validate_layer(layer)
        ts_key = self._resolve_ts_key(timestep)
        with h5py.File(self._path, "r") as f:
            return np.array(
                f[f"{ts_key}/attention/layer_{layer:02d}"], dtype=np.float32
            )

    def get_q_projections(
        self, timestep: int, layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read Q-projections: prefix [816, 8, 256] and suffix [51, 8, 256].

        Returns:
            Tuple of (q_prefix, q_suffix).
        """
        self._validate_layer(layer)
        ts_key = self._resolve_ts_key(timestep)
        base = f"{ts_key}/q_projections/layer_{layer:02d}"
        with h5py.File(self._path, "r") as f:
            q_prefix = np.array(f[f"{base}/prefix"], dtype=np.float32)
            q_suffix = np.array(f[f"{base}/suffix"], dtype=np.float32)
        return q_prefix, q_suffix

    def get_cmf_attended(
        self, timestep: int, layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read pre-computed CMF attended representations.

        Returns:
            Tuple of (attended_language [51, 8, 256],
                       attended_visual [51, 8, 256]).
        """
        self._validate_layer(layer)
        ts_key = self._resolve_ts_key(timestep)
        base = f"{ts_key}/cmf_attended/layer_{layer:02d}"
        with h5py.File(self._path, "r") as f:
            lang = np.array(f[f"{base}/language"], dtype=np.float32)
            vis = np.array(f[f"{base}/visual"], dtype=np.float32)
        return lang, vis

    def get_tsne(self, timestep: int, layer: int) -> np.ndarray:
        """Read t-SNE coordinates: float32 [867, 2]."""
        self._validate_layer(layer)
        ts_key = self._resolve_ts_key(timestep)
        with h5py.File(self._path, "r") as f:
            return np.array(
                f[f"{ts_key}/tsne/layer_{layer:02d}"], dtype=np.float32
            )

    def has_cmf_attended(self) -> bool:
        """Check whether this HDF5 file contains cmf_attended data."""
        keys = self.get_timestep_keys()
        if not keys:
            return False
        with h5py.File(self._path, "r") as f:
            return f"{keys[0]}/cmf_attended" in f

    @staticmethod
    def _validate_layer(layer: int) -> None:
        if layer not in SAMPLED_LAYERS:
            raise ValueError(
                f"Layer {layer} not in sampled layers {SAMPLED_LAYERS}"
            )


def scan_episodes(data_dir: Path) -> list[AnalyticsReader]:
    """Scan a directory for .h5 files and return readers sorted by name."""
    if not data_dir.is_dir():
        return []
    return [
        AnalyticsReader(p) for p in sorted(data_dir.glob("*.h5"))
    ]
