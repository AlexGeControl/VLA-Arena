"""Validate extracted HDF5 files against the InterpreT schema.

Checks:
  - All required groups/datasets exist with correct shapes and dtypes.
  - Attention rows sum to approximately 1.0.
  - t-SNE coordinates are finite.
  - Neighbor indices are within valid token ranges.
  - Metadata attributes are present and parseable.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

SAMPLED_LAYERS: list[int] = [0, 3, 6, 9, 12, 15, 17]
EXPECTED_TOTAL_TOKENS = 867
EXPECTED_SUFFIX_TOKENS = 51
EXPECTED_NUM_HEADS = 8
EXPECTED_NUM_ACTIONS = 50
EXPECTED_NUM_MODALITIES = 5
NEIGHBOR_DTYPE = np.dtype([("neighbor_index", "<i4"), ("distance", "<f4")])


class ValidationResult:
    """Accumulates pass/fail checks with messages."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.checks: list[tuple[bool, str]] = []

    def check(self, condition: bool, msg: str) -> None:
        self.checks.append((condition, msg))
        if not condition:
            logger.warning("FAIL [%s]: %s", self.filepath, msg)

    @property
    def passed(self) -> int:
        return sum(1 for ok, _ in self.checks if ok)

    @property
    def failed(self) -> int:
        return sum(1 for ok, _ in self.checks if not ok)

    def summary(self) -> str:
        total = len(self.checks)
        return f"{self.filepath}: {self.passed}/{total} passed, {self.failed} failed"


def validate_file(path: Path) -> ValidationResult:
    """Run all validation checks on a single HDF5 file.

    Args:
        path: Path to the .h5 file.

    Returns:
        ValidationResult with all check outcomes.
    """
    result = ValidationResult(path.name)

    with h5py.File(path, "r") as f:
        _validate_meta(f, result)
        _validate_cameras(f, result)
        _validate_timesteps(f, result)

    return result


def _validate_meta(f: h5py.File, result: ValidationResult) -> None:
    """Check /meta group and its attributes."""
    result.check("meta" in f, "/meta group exists")
    if "meta" not in f:
        return
    meta = f["meta"]
    for attr in ("episode_id", "task_instruction", "num_timesteps", "instruction_tokens", "sampled_layers"):
        result.check(attr in meta.attrs, f"/meta has attribute '{attr}'")

    if "sampled_layers" in meta.attrs:
        layers = json.loads(meta.attrs["sampled_layers"])
        result.check(layers == SAMPLED_LAYERS, f"sampled_layers == {SAMPLED_LAYERS}")

    if "instruction_tokens" in meta.attrs:
        tokens = json.loads(meta.attrs["instruction_tokens"])
        result.check(isinstance(tokens, list), "instruction_tokens is a list")


def _validate_cameras(f: h5py.File, result: ValidationResult) -> None:
    """Check /cameras group."""
    if "cameras" not in f:
        result.check(False, "/cameras group exists")
        return
    result.check(True, "/cameras group exists")
    for name in f["cameras"]:
        img = np.array(f[f"cameras/{name}"])
        result.check(img.dtype == np.uint8, f"cameras/{name} dtype is uint8")
        result.check(len(img.shape) == 3 and img.shape[2] == 3, f"cameras/{name} has 3 channels")


def _validate_timesteps(f: h5py.File, result: ValidationResult) -> None:
    """Check all timestep_NNN groups."""
    ts_groups = [k for k in f.keys() if k.startswith("timestep_")]
    result.check(len(ts_groups) > 0, "at least one timestep group exists")

    if "meta" in f:
        expected_count = int(f["meta"].attrs.get("num_timesteps", 0))
        result.check(
            len(ts_groups) == expected_count,
            f"num timestep groups ({len(ts_groups)}) == meta.num_timesteps ({expected_count})",
        )

    for ts_key in sorted(ts_groups):
        _validate_single_timestep(f, ts_key, result)


def _validate_single_timestep(f: h5py.File, ts_key: str, result: ValidationResult) -> None:
    """Validate datasets within a single timestep group."""
    _validate_token_meta(f, ts_key, result)
    _validate_attention_layers(f, ts_key, result)
    _validate_tsne_layers(f, ts_key, result)
    _validate_neighbor_layers(f, ts_key, result)


def _validate_token_meta(f: h5py.File, ts_key: str, result: ValidationResult) -> None:
    """Check token_meta JSON dataset."""
    key = f"{ts_key}/token_meta"
    result.check(key in f, f"{key} exists")
    if key not in f:
        return
    raw = f[key][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    tokens = json.loads(raw)
    result.check(isinstance(tokens, list), f"{key} parses as list")
    result.check(len(tokens) > 800, f"{key} has >800 entries (got {len(tokens)})")


def _validate_attention_layers(f: h5py.File, ts_key: str, result: ValidationResult) -> None:
    """Check attention/layer_XX datasets."""
    attn_grp = f"{ts_key}/attention"
    if attn_grp not in f:
        result.check(False, f"{attn_grp} group exists")
        return
    result.check(True, f"{attn_grp} group exists")

    for layer_idx in SAMPLED_LAYERS:
        ds_key = f"{attn_grp}/layer_{layer_idx:02d}"
        if ds_key not in f:
            result.check(False, f"{ds_key} exists")
            continue
        result.check(True, f"{ds_key} exists")
        arr = np.array(f[ds_key])
        result.check(arr.dtype == np.float32, f"{ds_key} dtype is float32")
        result.check(
            arr.shape[0] == EXPECTED_NUM_HEADS,
            f"{ds_key} shape[0]=={EXPECTED_NUM_HEADS} (got {arr.shape[0]})",
        )

        row_sums = arr.sum(axis=-1)
        mean_sum = float(row_sums.mean())
        result.check(
            0.95 < mean_sum < 1.05,
            f"{ds_key} attention rows sum to ~1.0 (mean={mean_sum:.4f})",
        )


def _validate_tsne_layers(f: h5py.File, ts_key: str, result: ValidationResult) -> None:
    """Check tsne/layer_XX datasets."""
    tsne_grp = f"{ts_key}/tsne"
    if tsne_grp not in f:
        result.check(False, f"{tsne_grp} group exists")
        return
    result.check(True, f"{tsne_grp} group exists")

    for layer_idx in SAMPLED_LAYERS:
        ds_key = f"{tsne_grp}/layer_{layer_idx:02d}"
        if ds_key not in f:
            result.check(False, f"{ds_key} exists")
            continue
        result.check(True, f"{ds_key} exists")
        arr = np.array(f[ds_key])
        result.check(arr.shape[1] == 2, f"{ds_key} shape[1]==2")
        result.check(np.all(np.isfinite(arr)), f"{ds_key} all values finite")


def _validate_neighbor_layers(f: h5py.File, ts_key: str, result: ValidationResult) -> None:
    """Check neighbors/layer_XX datasets."""
    nbr_grp = f"{ts_key}/neighbors"
    if nbr_grp not in f:
        result.check(False, f"{nbr_grp} group exists")
        return
    result.check(True, f"{nbr_grp} group exists")

    for layer_idx in SAMPLED_LAYERS:
        ds_key = f"{nbr_grp}/layer_{layer_idx:02d}"
        if ds_key not in f:
            result.check(False, f"{ds_key} exists")
            continue
        result.check(True, f"{ds_key} exists")
        arr = np.array(f[ds_key])
        result.check(
            arr.shape == (EXPECTED_NUM_ACTIONS, EXPECTED_NUM_MODALITIES),
            f"{ds_key} shape==(50,5) (got {arr.shape})",
        )
        indices = arr["neighbor_index"]
        result.check(
            np.all(indices >= 0) and np.all(indices < EXPECTED_TOTAL_TOKENS),
            f"{ds_key} neighbor indices in [0, {EXPECTED_TOTAL_TOKENS})",
        )


def main() -> None:
    """Validate all .h5 files in a directory (or a single file)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-h5-file-or-directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    paths = sorted(target.glob("*.h5")) if target.is_dir() else [target]

    if not paths:
        logger.error("No .h5 files found at %s", target)
        sys.exit(1)

    all_passed = True
    for p in paths:
        logger.info("Validating %s ...", p.name)
        result = validate_file(p)
        logger.info(result.summary())
        if result.failed > 0:
            all_passed = False
            for ok, msg in result.checks:
                if not ok:
                    logger.error("  FAIL: %s", msg)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
