#!/usr/bin/env python3
"""Three-modality EMAP evaluator with staged CPU–GPU pipeline.

Extends the original 2-modality EMAP (vision vs. language) to a 3-modality
decomposition (Vision × Language × State), computing all 6 non-trivial
marginals plus Shapley value attribution.

Architecture (see ``docs/pi-zero/emap/pipeline.md``):

  Stage 1  PLAN   — build a flat list of (anchor, donor, SwapMask) tuples
  Stage 2  BUILD  — materialise observations (parquet read + JPEG decode)
  Stage 3  INFER  — continuous GPU batch inference
  Stage 4  REDUCE — aggregate marginal means, compute Shapley values

Stages 2 and 3 overlap via a bounded producer–consumer queue so the
ThreadRipper (CPU) and Blackwell (GPU) stay continuously busy.

Design references:
  - docs/pi-zero/emap/marginalization.md  (3-modality marginal construction)
  - docs/pi-zero/emap/pipeline.md         (staged pipeline & SwapMask)

Usage:
    # dry-run
    python emap3_evaluator.py --cfg.num-mc-samples 2 --cfg.batch-size 4 \\
        --cfg.num-buckets 2 --cfg.output-dir ./emap3_dryrun

    # production (RTX PRO 6000 Blackwell, 96 GB)
    CUDA_VISIBLE_DEVICES=0 \\
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \\
    HF_HUB_DISABLE_XET=1 \\
    PYTHONUNBUFFERED=1 \\
    .venv/bin/python -u emap3_evaluator.py \\
        --cfg.num-mc-samples 50 --cfg.batch-size 200 \\
        --cfg.num-buckets 5 --cfg.output-dir ./emap3_production \\
        --cfg.model.device cuda:0 --cfg.num-build-workers 16
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
import pathlib
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Iterator

import numpy as np
import torch
import tqdm
import tyro

# ---------------------------------------------------------------------------
# Path setup — make sure openpi and vla_arena packages are importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ======================= SwapMask & InferenceSample ========================


class SwapMask(enum.IntFlag):
    """Bitmask controlling which modalities come from the donor vs. anchor.

    Bit set = take from donor; bit clear = take from anchor.
    See ``docs/pi-zero/emap/pipeline.md`` for the full mapping table.
    """

    NONE = 0b000
    SWAP_S = 0b001  # state from donor visuomotor sample
    SWAP_L = 0b010  # language from sampled instruction
    SWAP_V = 0b100  # vision from donor visuomotor sample


# Bimodal masks (popcount 1 — marginalize one modality)
MASK_VISION_STATE = SwapMask.SWAP_L
MASK_LANG_STATE = SwapMask.SWAP_V
MASK_LANG_VISION = SwapMask.SWAP_S

# Unimodal masks (popcount 2 — marginalize two modalities)
MASK_VISION_ONLY = SwapMask.SWAP_L | SwapMask.SWAP_S
MASK_LANG_ONLY = SwapMask.SWAP_V | SwapMask.SWAP_S
MASK_STATE_ONLY = SwapMask.SWAP_V | SwapMask.SWAP_L

ALL_MARGINAL_MASKS: list[SwapMask] = [
    MASK_VISION_STATE,
    MASK_LANG_STATE,
    MASK_LANG_VISION,
    MASK_VISION_ONLY,
    MASK_LANG_ONLY,
    MASK_STATE_ONLY,
]

MARGINAL_NAMES: dict[SwapMask, str] = {
    MASK_VISION_STATE: "vision_state",
    MASK_LANG_STATE: "lang_state",
    MASK_LANG_VISION: "lang_vision",
    MASK_VISION_ONLY: "vision_only",
    MASK_LANG_ONLY: "lang_only",
    MASK_STATE_ONLY: "state_only",
}


@dataclasses.dataclass(slots=True)
class InferenceSample:
    """Lightweight descriptor for one forward pass — no image data, just indices."""

    anchor_idx: int
    anchor_ep: int
    anchor_row: int
    donor_ep: int
    donor_row: int
    mask: SwapMask


# ============================= Configuration ==============================


class ActionHead(enum.Enum):
    """Action trajectory generation strategy."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"


_ACTION_HEAD_DEFAULTS: dict[ActionHead, tuple[str, str]] = {
    ActionHead.PI0: (
        "pi0_vla_arena_low_mem_finetune",
        "VLA-Arena/pi0-vla-arena-fintuned",
    ),
    ActionHead.PI0_FAST: (
        "pi0_fast_vla_arena_low_mem_finetune",
        "VLA-Arena/pi0-fast-vla-arena-fintuned",
    ),
}


@dataclasses.dataclass
class ModelConfig:
    """Model selection and checkpoint configuration."""

    action_head: ActionHead = ActionHead.PI0
    config_name: str = ""
    checkpoint_repo: str = ""
    device: str = "cuda:0"

    def __post_init__(self):
        default_config, default_repo = _ACTION_HEAD_DEFAULTS[self.action_head]
        if not self.config_name:
            self.config_name = default_config
        if not self.checkpoint_repo:
            self.checkpoint_repo = default_repo
        if not self.checkpoint_repo:
            raise ValueError(
                f"No default checkpoint for {self.action_head.value}. "
                "Provide --cfg.model.checkpoint-repo explicitly."
            )


@dataclasses.dataclass
class EMAP3Config:
    """Configuration for the 3-modality EMAP evaluator.

    tyro turns each field into a CLI flag (``--cfg.<field-name>``).
    """

    # ----- Model -----
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    # ----- Dataset -----
    dataset_repo_id: str = "VLA-Arena/VLA_Arena_L0_S_lerobot_openpi"

    # ----- EMAP parameters -----
    num_mc_samples: int = 50
    num_buckets: int = 5
    eval_horizon: int = 10
    action_dims: int = 7

    # ----- Batch inference -----
    batch_size: int = 200

    # ----- Pipeline -----
    num_build_workers: int = 8
    prefetch_batches: int = 2

    # ----- I/O -----
    output_dir: str = "./emap3_results"
    seed: int = 42


# ========================= Batch Inference Helpers ========================


def _collate_to_jax(samples: list[dict]) -> dict:
    """Stack per-sample dicts into a single batched dict of JAX arrays."""
    batched: dict = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if isinstance(vals[0], dict):
            batched[key] = {
                k: jnp.asarray(np.stack([v[k] for v in vals]))
                for k in vals[0]
            }
        elif isinstance(vals[0], np.ndarray):
            batched[key] = jnp.asarray(np.stack(vals))
        else:
            batched[key] = jnp.asarray(np.array(vals))
    return batched


def _collate_to_pytorch(samples: list[dict], device: str) -> dict:
    """Stack per-sample dicts into a single batched dict of PyTorch tensors."""
    batched: dict = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if isinstance(vals[0], dict):
            batched[key] = {
                k: torch.from_numpy(np.stack([v[k] for v in vals])).to(device)
                for k in vals[0]
            }
        elif isinstance(vals[0], np.ndarray):
            batched[key] = torch.from_numpy(np.stack(vals)).to(device)
        else:
            batched[key] = torch.tensor(np.array(vals), device=device)
    return batched


def batch_infer(
    policy: _policy.Policy,
    obs_list: list[dict],
    batch_size: int = 200,
) -> list[dict]:
    """Run batched inference, padding to fixed ``batch_size`` for JIT stability."""
    all_results: list[dict] = []
    is_pytorch = getattr(policy, "_is_pytorch_model", False)

    for start in range(0, len(obs_list), batch_size):
        chunk = obs_list[start : start + batch_size]
        actual_size = len(chunk)

        while len(chunk) < batch_size:
            chunk.append(chunk[0])

        transformed = []
        for obs in chunk:
            inp = jax.tree.map(lambda x: x, obs)
            inp = policy._input_transform(inp)
            transformed.append(inp)

        if is_pytorch:
            device = policy._pytorch_device
            batched = _collate_to_pytorch(transformed, device)
            observation = _model.Observation.from_dict(batched)
            actions = policy._sample_actions(device, observation)

            for j in range(actual_size):
                outputs = {
                    "state": batched["state"][j].detach().cpu().numpy(),
                    "actions": actions[j].detach().cpu().numpy(),
                }
                outputs = policy._output_transform(outputs)
                all_results.append(outputs)
        else:
            batched = _collate_to_jax(transformed)
            observation = _model.Observation.from_dict(batched)
            policy._rng, sample_rng = jax.random.split(policy._rng)
            actions = policy._sample_actions(
                sample_rng, observation, **policy._sample_kwargs
            )

            for j in range(actual_size):
                outputs = {
                    "state": np.asarray(batched["state"][j]),
                    "actions": np.asarray(actions[j]),
                }
                outputs = policy._output_transform(outputs)
                all_results.append(outputs)

    return all_results


# ============================== Model Loading =============================


def discover_norm_stats(
    checkpoint_dir: str | pathlib.Path,
) -> dict | None:
    """Search for norm_stats.json anywhere under ``<checkpoint>/assets/``."""
    from openpi.shared import normalize as _normalize

    assets_dir = pathlib.Path(checkpoint_dir) / "assets"
    if assets_dir.exists():
        for ns_path in assets_dir.rglob("norm_stats.json"):
            logger.info(f"Found norm_stats at: {ns_path}")
            return _normalize.load(ns_path.parent)

    logger.warning(
        "No norm_stats.json found — "
        "normalization will rely on data config defaults."
    )
    return None


def load_policy(cfg: EMAP3Config) -> _policy.Policy:
    """Download checkpoint and build a ``Policy`` for local inference."""
    from huggingface_hub import snapshot_download, logging as hf_logging

    hf_logging.set_verbosity_debug()
    logger.info(f"Downloading checkpoint from {cfg.model.checkpoint_repo} ...")
    checkpoint_dir = snapshot_download(cfg.model.checkpoint_repo)
    logger.info(f"Checkpoint at: {checkpoint_dir}")
    hf_logging.set_verbosity_warning()

    norm_stats = discover_norm_stats(checkpoint_dir)
    train_config = _config.get_config(cfg.model.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device=cfg.model.device,
        norm_stats=norm_stats,
    )
    logger.info(f"Policy loaded on {cfg.model.device}")
    return policy


# ============================= Data Loading ===============================


def decode_image_bytes(raw: dict) -> np.ndarray:
    """Decode a LeRobot serialized image column into a numpy array."""
    import io
    from PIL import Image as PILImage

    return np.asarray(PILImage.open(io.BytesIO(raw["bytes"])))


def resolve_dataset_root(repo_id: str) -> tuple[str | None, pathlib.Path]:
    """Resolve the on-disk root for a VLA-Arena LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    default_root = HF_LEROBOT_HOME / repo_id
    nested_root = default_root / "VLA_Arena"
    if (nested_root / "meta" / "info.json").exists():
        return str(nested_root), nested_root
    return None, default_root


class ParquetDataset:
    """Lightweight random-access dataset backed by parquet files.

    ``dataset[i]`` returns the row at global index ``i`` as a plain dict.
    Image columns are decoded lazily via ``decode_image_bytes``.
    """

    def __init__(self, data_dir: str | pathlib.Path):
        import pyarrow.parquet as pq

        data_dir = pathlib.Path(data_dir)
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files under {data_dir}")
        logger.info(f"Loading {len(parquet_files)} parquet files from {data_dir}")
        self._table = pq.read_table(parquet_files)
        logger.info(f"Loaded {len(self._table)} rows")

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx: int) -> dict:
        row = {
            col: self._table.column(col)[idx].as_py()
            for col in self._table.column_names
        }
        for img_key in ("image", "wrist_image"):
            if img_key in row and isinstance(row[img_key], dict) and "bytes" in row[img_key]:
                row[img_key] = decode_image_bytes(row[img_key])
        return row

    def get_raw(self, idx: int) -> dict:
        """Return a row *without* decoding images (for instruction-only swaps)."""
        return {
            col: self._table.column(col)[idx].as_py()
            for col in self._table.column_names
        }


def load_dataset(cfg: EMAP3Config):
    """Load the LeRobot dataset and return (dataset, metadata)."""
    logger.info(f"Loading dataset metadata from {cfg.dataset_repo_id} ...")
    root, data_root = resolve_dataset_root(cfg.dataset_repo_id)
    meta = lerobot_dataset.LeRobotDatasetMetadata(
        cfg.dataset_repo_id, root=root, revision="main",
    )
    dataset = ParquetDataset(data_root / "data")
    logger.info(
        f"Dataset loaded: {len(dataset)} frames, "
        f"{meta.total_episodes} episodes"
    )
    return dataset, meta


# ==================== Progress-Stratified Sampling ========================


def _get_episode_boundaries(meta) -> tuple[list[int], list[int]]:
    """Compute ``[ep_from, ep_to)`` global-row-index intervals from episode metadata."""
    ep_from: list[int] = []
    ep_to: list[int] = []
    cumulative = 0
    for ep_idx in sorted(meta.episodes.keys()):
        length = meta.episodes[ep_idx]["length"]
        ep_from.append(cumulative)
        ep_to.append(cumulative + length)
        cumulative += length
    return ep_from, ep_to


def compute_sample_plan(
    meta,
    num_buckets: int = 5,
    margin: int = 10,
) -> dict:
    """Compute progress-stratified sampling plan.

    Returns dict with ``bucket_samples`` (list of lists of ``(ep_idx, row_idx)``
    tuples) and ``skipped_episodes``.
    """
    ep_from, ep_to = _get_episode_boundaries(meta)
    bucket_fractions = np.linspace(0.0, 1.0, num_buckets)
    bucket_samples: list[list[tuple[int, int]]] = [[] for _ in range(num_buckets)]
    skipped: list[int] = []

    for ep_idx in range(len(ep_from)):
        start, end = int(ep_from[ep_idx]), int(ep_to[ep_idx])
        if end - start < 2 * margin + 1:
            skipped.append(ep_idx)
            continue

        valid_start = start + margin
        valid_len = (end - margin) - valid_start

        for b_idx, frac in enumerate(bucket_fractions):
            frame_offset = int(round(frac * (valid_len - 1)))
            bucket_samples[b_idx].append((ep_idx, valid_start + frame_offset))

    return {"bucket_samples": bucket_samples, "skipped_episodes": skipped}


def build_task_map(meta) -> dict[int, str]:
    """Return ``{episode_idx: instruction_string}`` from dataset metadata."""
    return {
        ep_idx: ep_data["tasks"][0]
        for ep_idx, ep_data in meta.episodes.items()
    }


def extract_delta_actions(
    result: dict,
    obs: dict,
    eval_horizon: int = 10,
) -> np.ndarray:
    """Convert absolute-pose policy output to a flat 70-D delta-action vector."""
    actions = result["actions"][:eval_horizon, :]
    delta = actions.copy()
    current_state = np.asarray(obs["observation/state"], dtype=np.float32)
    delta[:, :6] -= current_state[:6]
    return delta.reshape(-1)


# ======================== Stage 1: Inference Plan =========================


def build_inference_plan(
    bucket_samples: list[tuple[int, int]],
    ep_instructions: dict[int, str],
    rng: np.random.Generator,
    N: int,
) -> tuple[list[InferenceSample], list[InferenceSample]]:
    """Build the full inference plan for one progress bucket.

    Returns:
        (f_full_plan, marginal_plan) — two flat lists of ``InferenceSample``.
        ``f_full_plan`` contains one sample per anchor (mask=NONE).
        ``marginal_plan`` contains 6*N samples per anchor (all 6 marginal types).
    """
    all_instructions = list(set(ep_instructions.values()))
    all_ep_by_instr: dict[str, list[int]] = {}
    for ep, instr in ep_instructions.items():
        all_ep_by_instr.setdefault(instr, []).append(ep)

    f_full_plan: list[InferenceSample] = []
    marginal_plan: list[InferenceSample] = []

    donor_pool = bucket_samples

    for anchor_idx, (anchor_ep, anchor_row) in enumerate(bucket_samples):
        # -- f_full --
        f_full_plan.append(InferenceSample(
            anchor_idx=anchor_idx,
            anchor_ep=anchor_ep,
            anchor_row=anchor_row,
            donor_ep=anchor_ep,
            donor_row=anchor_row,
            mask=SwapMask.NONE,
        ))

        # -- Sample N global instructions (self-exclusion) --
        eligible_instrs = [
            instr for instr in all_instructions
            if instr != ep_instructions[anchor_ep]
        ]
        if not eligible_instrs:
            eligible_instrs = all_instructions
        sampled_instrs = rng.choice(eligible_instrs, size=N, replace=True)
        instr_donor_eps: list[int] = []
        for instr in sampled_instrs:
            candidates = all_ep_by_instr[instr]
            instr_donor_eps.append(int(rng.choice(candidates)))

        # -- Sample N donor visuomotor observations (self-exclusion) --
        eligible_donors = [(e, r) for (e, r) in donor_pool if e != anchor_ep]
        if len(eligible_donors) < N:
            donor_indices = rng.choice(len(eligible_donors), size=N, replace=True)
        else:
            donor_indices = rng.choice(len(eligible_donors), size=N, replace=False)
        sampled_donors = [eligible_donors[int(i)] for i in donor_indices]

        # -- Emit 6 marginal types × N samples --
        for i in range(N):
            d_ep_vis, d_row_vis = sampled_donors[i]
            d_ep_instr = instr_donor_eps[i]

            for mask in ALL_MARGINAL_MASKS:
                donor_ep = d_ep_instr if (mask & SwapMask.SWAP_L) else anchor_ep
                donor_row = d_row_vis if (mask & (SwapMask.SWAP_V | SwapMask.SWAP_S)) else anchor_row

                marginal_plan.append(InferenceSample(
                    anchor_idx=anchor_idx,
                    anchor_ep=anchor_ep,
                    anchor_row=anchor_row,
                    donor_ep=donor_ep,
                    donor_row=donor_row,
                    mask=mask,
                ))

    return f_full_plan, marginal_plan


# =================== Stage 2: Observation Builder =========================


class ObservationBuilder:
    """Builds observation dicts from ``InferenceSample`` descriptors.

    Wraps a ``ParquetDataset`` and an LRU cache of decoded rows to avoid
    redundant parquet reads and JPEG decodes.
    """

    def __init__(
        self,
        dataset: ParquetDataset,
        ep_instructions: dict[int, str],
        cache_size: int = 2048,
    ):
        self._dataset = dataset
        self._ep_instructions = ep_instructions

        @lru_cache(maxsize=cache_size)
        def _cached_row(row_idx: int) -> dict:
            return dataset[row_idx]

        self._cached_row = _cached_row

    def build(self, sample: InferenceSample) -> dict:
        """Materialise an ``InferenceSample`` into a policy-ready observation dict."""
        mask = sample.mask
        needs_donor_vis = bool(mask & (SwapMask.SWAP_V | SwapMask.SWAP_S))

        anchor_data = self._cached_row(sample.anchor_row)
        donor_data = self._cached_row(sample.donor_row) if needs_donor_vis else None

        vis_src = donor_data if (mask & SwapMask.SWAP_V) else anchor_data
        state_src = donor_data if (mask & SwapMask.SWAP_S) else anchor_data
        prompt = (
            self._ep_instructions[sample.donor_ep]
            if (mask & SwapMask.SWAP_L)
            else self._ep_instructions[sample.anchor_ep]
        )

        return {
            "observation/image": np.asarray(vis_src["image"]),
            "observation/wrist_image": np.asarray(vis_src["wrist_image"]),
            "observation/state": np.asarray(state_src["state"], dtype=np.float32),
            "prompt": prompt,
        }


def _prefetch_batches(
    plan: list[InferenceSample],
    builder: ObservationBuilder,
    batch_size: int,
    num_workers: int,
    prefetch_depth: int,
    out_queue: queue.Queue,
):
    """Producer thread: build observations in parallel and enqueue batches.

    Each item placed on ``out_queue`` is a list of ``(InferenceSample, obs_dict)``
    tuples forming one GPU batch.  A sentinel ``None`` signals completion.
    """
    executor = ThreadPoolExecutor(max_workers=num_workers)

    for start in range(0, len(plan), batch_size):
        chunk = plan[start : start + batch_size]
        futures = [executor.submit(builder.build, s) for s in chunk]
        batch = [(s, f.result()) for s, f in zip(chunk, futures)]
        out_queue.put(batch)

    executor.shutdown(wait=True)
    out_queue.put(None)


# =================== Stage 3: GPU Inference Loop ==========================


def run_inference_pipeline(
    plan: list[InferenceSample],
    builder: ObservationBuilder,
    policy: _policy.Policy,
    batch_size: int,
    eval_horizon: int,
    num_workers: int,
    prefetch_depth: int,
    desc: str = "",
) -> list[tuple[int, SwapMask, np.ndarray]]:
    """Execute the producer–consumer inference pipeline.

    Returns a list of ``(anchor_idx, mask, delta_action_70d)`` tuples.
    """
    if not plan:
        return []

    obs_queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)

    producer = threading.Thread(
        target=_prefetch_batches,
        args=(plan, builder, batch_size, num_workers, prefetch_depth, obs_queue),
        daemon=True,
    )
    producer.start()

    results: list[tuple[int, SwapMask, np.ndarray]] = []
    total_batches = (len(plan) + batch_size - 1) // batch_size
    pbar = tqdm.tqdm(total=total_batches, desc=desc, leave=True)

    while True:
        batch = obs_queue.get()
        if batch is None:
            break

        samples_meta = [s for s, _ in batch]
        obs_list = [obs for _, obs in batch]

        infer_results = batch_infer(policy, obs_list, batch_size=batch_size)

        for i, (meta, obs) in enumerate(zip(samples_meta, obs_list)):
            delta = extract_delta_actions(infer_results[i], obs, eval_horizon)
            results.append((meta.anchor_idx, meta.mask, delta))

        pbar.update(1)

    pbar.close()
    producer.join()
    return results


# ====================== Stage 4: Reduce & Metrics ========================


def _explained_variance(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    """Compute R² and residual ratio for a projection against ground truth.

    Returns:
        ``r_squared``:     fraction of variance **explained** (higher = better).
        ``residual_ratio``: fraction of variance **unexplained** (lower = better).
        The two always sum to 1.
    """
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - baseline) ** 2))
    residual_ratio = ss_res / (ss_tot + 1e-12)
    return {"r_squared": 1.0 - residual_ratio, "residual_ratio": residual_ratio}


def _shapley_values(
    coalition_values: dict[frozenset[str], np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute per-sample Shapley values from coalition value arrays.

    Args:
        coalition_values: Maps frozenset of kept modalities to an array of
            shape ``(N, D)`` representing the coalition's per-sample predictions.
            Must include all 8 coalitions (empty set through {V, L, S}).

    Returns:
        ``{modality: shapley_array}`` where each array is ``(N, D)``.
        φ(V) + φ(L) + φ(S) = f_full - μ  (efficiency axiom).
    """
    players = ["V", "L", "S"]
    n_players = len(players)
    # factorial(n_players) = 6
    from math import factorial
    n_fact = factorial(n_players)

    # Enumerate all permutations
    from itertools import permutations
    all_perms = list(permutations(players))

    shapley = {}
    for p in players:
        contrib = np.zeros_like(coalition_values[frozenset()])
        for perm in all_perms:
            # S_π = set of players before p in this permutation
            idx = list(perm).index(p)
            before = frozenset(perm[:idx])
            with_p = before | {p}
            contrib += coalition_values[with_p] - coalition_values[before]
        shapley[p] = contrib / n_fact

    return shapley


def reduce_results(
    f_full_raw: list[tuple[int, SwapMask, np.ndarray]],
    marginal_raw: list[tuple[int, SwapMask, np.ndarray]],
    num_anchors: int,
    cfg: EMAP3Config,
) -> dict:
    """Aggregate raw inference results into per-anchor marginal means and metrics.

    Returns a dict with all marginal arrays, Shapley values, R² metrics,
    and per-bucket breakdowns.
    """
    D = cfg.eval_horizon * cfg.action_dims

    # -- Collect f_full --
    f_full = np.zeros((num_anchors, D), dtype=np.float64)
    for anchor_idx, _, delta in f_full_raw:
        f_full[anchor_idx] = delta

    # -- Accumulate marginal sums --
    marginal_sums: dict[SwapMask, np.ndarray] = {
        m: np.zeros((num_anchors, D), dtype=np.float64) for m in ALL_MARGINAL_MASKS
    }
    marginal_counts: dict[SwapMask, np.ndarray] = {
        m: np.zeros(num_anchors, dtype=np.int32) for m in ALL_MARGINAL_MASKS
    }

    for anchor_idx, mask, delta in marginal_raw:
        marginal_sums[mask][anchor_idx] += delta
        marginal_counts[mask][anchor_idx] += 1

    marginal_means: dict[SwapMask, np.ndarray] = {}
    for m in ALL_MARGINAL_MASKS:
        counts = marginal_counts[m][:, None].clip(min=1)
        marginal_means[m] = marginal_sums[m] / counts

    N = num_anchors

    # -- Z-score normalisation (per-dimension, based on f_full) --
    mu_norm = f_full.mean(axis=0)
    std_norm = f_full.std(axis=0) + 1e-8

    f_z = (f_full - mu_norm) / std_norm
    m_z: dict[SwapMask, np.ndarray] = {
        m: (marginal_means[m] - mu_norm) / std_norm for m in ALL_MARGINAL_MASKS
    }

    mu = f_z.mean(axis=0)

    # -- Build coalition value arrays for Shapley computation --
    # Map each coalition (frozenset of KEPT modalities) to its z-scored prediction
    coalition_values: dict[frozenset[str], np.ndarray] = {
        frozenset(): np.tile(mu, (N, 1)),  # empty coalition → grand mean
        frozenset({"V", "L", "S"}): f_z,   # full coalition → f_full
        # Bimodal (keep 2)
        frozenset({"V", "S"}): m_z[MASK_VISION_STATE],
        frozenset({"L", "S"}): m_z[MASK_LANG_STATE],
        frozenset({"L", "V"}): m_z[MASK_LANG_VISION],
        # Unimodal (keep 1)
        frozenset({"V"}): m_z[MASK_VISION_ONLY],
        frozenset({"L"}): m_z[MASK_LANG_ONLY],
        frozenset({"S"}): m_z[MASK_STATE_ONLY],
    }

    # -- Shapley values --
    shapley = _shapley_values(coalition_values)

    # -- Additive projection: g_V(v) + g_L(l) + g_S(s) + μ --
    g_V = m_z[MASK_VISION_ONLY] - mu
    g_L = m_z[MASK_LANG_ONLY] - mu
    g_S = m_z[MASK_STATE_ONLY] - mu
    f_additive = g_V + g_L + g_S + mu

    # -- Global R² metrics --
    metrics: dict[str, dict] = {}
    metrics["additive_3way"] = _explained_variance(f_z, f_additive, mu)
    for m in ALL_MARGINAL_MASKS:
        metrics[MARGINAL_NAMES[m]] = _explained_variance(f_z, m_z[m], mu)

    # Log summary
    logger.info("--- Global R² ---")
    for label, vals in metrics.items():
        logger.info(f"  {label:20s}  R²={vals['r_squared']:.4f}")

    shapley_global = {
        p: float(np.mean(np.sum(shapley[p] ** 2, axis=1)))
        for p in ["V", "L", "S"]
    }
    logger.info(f"  Shapley mean |φ|²:  V={shapley_global['V']:.4f}  "
                f"L={shapley_global['L']:.4f}  S={shapley_global['S']:.4f}")

    # -- Per-bucket breakdown --
    bucket_size = N // cfg.num_buckets
    per_bucket: list[dict] = []
    for b in range(cfg.num_buckets):
        s = b * bucket_size
        e = s + bucket_size if b < cfg.num_buckets - 1 else N
        b_f = f_z[s:e]

        bucket_metrics: dict[str, dict] = {}
        bucket_metrics["additive_3way"] = _explained_variance(
            b_f, f_additive[s:e], mu,
        )
        for m in ALL_MARGINAL_MASKS:
            bucket_metrics[MARGINAL_NAMES[m]] = _explained_variance(
                b_f, m_z[m][s:e], mu,
            )

        bucket_shapley = {
            p: float(np.mean(np.sum(shapley[p][s:e] ** 2, axis=1)))
            for p in ["V", "L", "S"]
        }

        per_bucket.append({
            "bucket": b,
            "progress_pct": f"{b / max(cfg.num_buckets - 1, 1) * 100:.0f}%",
            "num_samples": e - s,
            "metrics": bucket_metrics,
            "shapley_mean_sq": bucket_shapley,
        })

    return {
        "global_metrics": metrics,
        "shapley_global_mean_sq": shapley_global,
        "per_bucket": per_bucket,
        "num_samples": N,
        "action_dim": D,
        "config": {
            "num_mc_samples": cfg.num_mc_samples,
            "num_buckets": cfg.num_buckets,
            "eval_horizon": cfg.eval_horizon,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        },
        "normalization": {
            "mean": mu_norm.tolist(),
            "std": std_norm.tolist(),
        },
    }


# ================================= Main ===================================


def main(cfg: EMAP3Config) -> None:
    """Three-modality EMAP evaluator entry point."""
    logger.info("=" * 60)
    logger.info("EMAP3 — Three-Modality Evaluator for Pi-Zero")
    logger.info("=" * 60)
    logger.info(f"Config: {cfg}")

    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "emap3_config.json"
    config_path.write_text(json.dumps(
        dataclasses.asdict(cfg),
        indent=2,
        default=lambda o: o.value if isinstance(o, enum.Enum) else str(o),
    ))
    logger.info(f"Config saved to {config_path}")

    # ---- 1. Load model ----
    logger.info("----- Loading model -----")
    policy = load_policy(cfg)

    # ---- 2. Load dataset ----
    logger.info("----- Loading dataset -----")
    dataset, meta = load_dataset(cfg)
    ep_instructions = build_task_map(meta)

    # ---- 3. Sampling plan ----
    logger.info("----- Computing sampling plan -----")
    plan = compute_sample_plan(meta, cfg.num_buckets, margin=cfg.eval_horizon)
    bucket_samples = plan["bucket_samples"]
    total_anchors = sum(len(b) for b in bucket_samples)
    logger.info(
        f"Sampling plan: {cfg.num_buckets} buckets, "
        f"{total_anchors} anchors, "
        f"{len(plan['skipped_episodes'])} episodes skipped"
    )

    # ---- 4. JIT warmup ----
    logger.info("----- JIT warmup -----")
    first_ep, first_row = bucket_samples[0][0]
    warmup_obs = {
        "observation/image": np.asarray(dataset[first_row]["image"]),
        "observation/wrist_image": np.asarray(dataset[first_row]["wrist_image"]),
        "observation/state": np.asarray(dataset[first_row]["state"], dtype=np.float32),
        "prompt": ep_instructions[first_ep],
    }
    warmup_t0 = time.time()
    batch_infer(policy, [warmup_obs] * cfg.batch_size, batch_size=cfg.batch_size)
    logger.info(f"JIT warmup done in {time.time() - warmup_t0:.1f}s")

    # ---- 5. Per-bucket pipeline ----
    builder = ObservationBuilder(dataset, ep_instructions)
    rng = np.random.default_rng(cfg.seed)

    all_f_full_raw: list[tuple[int, SwapMask, np.ndarray]] = []
    all_marginal_raw: list[tuple[int, SwapMask, np.ndarray]] = []
    anchor_offset = 0
    wall_start = time.time()

    for b_idx in range(cfg.num_buckets):
        bucket = bucket_samples[b_idx]
        tag = f"bucket_{b_idx}"
        ckpt_path = output_dir / f"checkpoint_{tag}.pt"

        if ckpt_path.exists():
            logger.info(f"[{tag}] Resuming from checkpoint")
            ckpt = torch.load(ckpt_path, weights_only=False)
            all_f_full_raw.extend(ckpt["f_full_raw"])
            all_marginal_raw.extend(ckpt["marginal_raw"])
            anchor_offset += len(bucket)
            continue

        logger.info(f"[{tag}] Planning {len(bucket)} anchors × "
                     f"6 marginals × N={cfg.num_mc_samples} ...")
        f_full_plan, marginal_plan = build_inference_plan(
            bucket, ep_instructions, rng, cfg.num_mc_samples,
        )

        # Shift anchor indices by offset so global arrays are contiguous
        for s in f_full_plan:
            s.anchor_idx += anchor_offset
        for s in marginal_plan:
            s.anchor_idx += anchor_offset

        logger.info(f"[{tag}] Plan: {len(f_full_plan)} f_full + "
                     f"{len(marginal_plan)} marginals = "
                     f"{len(f_full_plan) + len(marginal_plan)} total passes")

        # -- f_full pass (small, no overlap needed) --
        logger.info(f"[{tag}] Running f_full inference ...")
        bucket_f_full = run_inference_pipeline(
            f_full_plan, builder, policy, cfg.batch_size,
            cfg.eval_horizon, cfg.num_build_workers,
            cfg.prefetch_batches, desc=f"{tag}/f_full",
        )

        # -- Marginal passes (large, producer–consumer) --
        logger.info(f"[{tag}] Running marginal inference ...")
        bucket_marginals = run_inference_pipeline(
            marginal_plan, builder, policy, cfg.batch_size,
            cfg.eval_horizon, cfg.num_build_workers,
            cfg.prefetch_batches, desc=f"{tag}/marginals",
        )

        # -- Checkpoint --
        torch.save({
            "f_full_raw": bucket_f_full,
            "marginal_raw": bucket_marginals,
            "bucket_idx": b_idx,
            "num_anchors": len(bucket),
        }, ckpt_path)
        logger.info(f"[{tag}] Checkpoint saved ({len(bucket)} anchors, "
                     f"{len(bucket_f_full) + len(bucket_marginals)} passes)")

        all_f_full_raw.extend(bucket_f_full)
        all_marginal_raw.extend(bucket_marginals)
        anchor_offset += len(bucket)

        # Clear LRU cache between buckets to free memory
        builder._cached_row.cache_clear()

    wall_elapsed = time.time() - wall_start
    total_passes = len(all_f_full_raw) + len(all_marginal_raw)
    logger.info(
        f"Inference complete: {total_passes} passes in {wall_elapsed:.1f}s "
        f"({total_passes / max(wall_elapsed, 1e-9):.1f} passes/s)"
    )

    # ---- 6. Reduce ----
    logger.info("----- Computing metrics -----")
    summary = reduce_results(all_f_full_raw, all_marginal_raw, anchor_offset, cfg)

    results_path = output_dir / "emap3_results.json"
    results_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Results saved to {results_path}")

    # Save raw vectors for downstream analysis
    raw_path = output_dir / "emap3_raw_vectors.pt"
    f_full_arr = np.zeros((anchor_offset, cfg.eval_horizon * cfg.action_dims))
    for idx, _, delta in all_f_full_raw:
        f_full_arr[idx] = delta
    marginal_arrays: dict[str, np.ndarray] = {}
    for m in ALL_MARGINAL_MASKS:
        name = MARGINAL_NAMES[m]
        sums = np.zeros((anchor_offset, cfg.eval_horizon * cfg.action_dims))
        counts = np.zeros(anchor_offset)
        for idx, mask, delta in all_marginal_raw:
            if mask == m:
                sums[idx] += delta
                counts[idx] += 1
        counts = counts.clip(min=1)
        marginal_arrays[name] = sums / counts[:, None]

    torch.save({"f_full": f_full_arr, **marginal_arrays}, raw_path)
    logger.info(f"Raw vectors saved to {raw_path}")

    # ---- Print summary ----
    logger.info("=" * 60)
    logger.info(f"Samples: {summary['num_samples']}  Action dim: {summary['action_dim']}")
    logger.info("--- Global R² ---")
    for label, vals in summary["global_metrics"].items():
        logger.info(f"  {label:20s}  R²={vals['r_squared']:.4f}  "
                     f"residual={vals['residual_ratio']:.4f}")
    sv = summary["shapley_global_mean_sq"]
    logger.info(f"  Shapley |φ|²:       V={sv['V']:.4f}  L={sv['L']:.4f}  S={sv['S']:.4f}")
    logger.info("--- Per-bucket ---")
    for b in summary["per_bucket"]:
        add = b["metrics"]["additive_3way"]
        sh = b["shapley_mean_sq"]
        logger.info(
            f"  Bucket {b['bucket']} ({b['progress_pct']}): "
            f"add_R²={add['r_squared']:.3f}  "
            f"φ²: V={sh['V']:.3f} L={sh['L']:.3f} S={sh['S']:.3f}  "
            f"N={b['num_samples']}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
