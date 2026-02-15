#!/usr/bin/env python3
"""EMAP (Empirical Multimodally-Additive function Projection) Evaluator for Pi-Zero on VLA-Arena.

Measures the strength of multimodal interaction in the fine-tuned Pi-Zero model by computing
how much the model's action predictions depend on the *interaction* between vision and language,
versus exploiting each modality independently.

Reference: Hessel & Lee, "Does My Multimodal Model Learn Cross-modal Interactions?
           It's Harder to Tell than You Might Think!" (EMNLP 2020)
           https://arxiv.org/abs/2010.06572

Usage:
    python emap_evaluator.py --output_dir ./emap_results
    python emap_evaluator.py --K 2 --M 2 --batch_size 2  # dry-run
    CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python -u emap_evaluator.py \
    --cfg.K 50 \
    --cfg.M 50 \
    --cfg.num-buckets 5 \
    --cfg.output-dir ./emap_production_K50_M50 \
    --cfg.device cuda:0
"""

import dataclasses
import json
import logging
import os
import pathlib
import sys
import time

import jax
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


# ============================= Configuration ==============================


@dataclasses.dataclass
class EMAPConfig:
    """Configuration for the EMAP evaluator."""

    # ----- Model -----
    checkpoint_repo: str = "VLA-Arena/pi0-vla-arena-fintuned"
    config_name: str = "pi0_vla_arena_low_mem_finetune"
    device: str = "cuda:0"

    # ----- Dataset -----
    dataset_repo_id: str = "VLA-Arena/VLA_Arena_L0_S_lerobot_openpi"

    # ----- EMAP parameters -----
    K: int = 50  # Number of random instructions for visual marginals
    M: int = 50  # Number of random (image, state) pairs for language marginals
    num_buckets: int = 5  # Progress-stratified buckets per episode
    eval_horizon: int = 10  # Action steps to keep for EMAP (70D vector)
    action_dims: int = 7  # Dimensions per action step

    # ----- Batch inference -----
    batch_size: int = 50  # Batch size (matches K=M for compile consistency)
    auto_batch: bool = False  # If True, auto-tune batch size at startup

    # ----- I/O -----
    output_dir: str = "./emap_results"
    seed: int = 42


# ========================= Batch Inference Helpers ========================


def _collate_transformed(samples: list[dict], device: str) -> dict:
    """Stack a list of per-sample transformed dicts into a single batched dict of tensors.

    Handles nested dicts (e.g. ``image``, ``image_mask``) and flat arrays.
    """
    batched: dict = {}
    keys = samples[0].keys()
    for key in keys:
        vals = [s[key] for s in samples]
        if isinstance(vals[0], dict):
            batched[key] = {
                k: torch.from_numpy(np.stack([v[k] for v in vals])).to(device)
                for k in vals[0]
            }
        elif isinstance(vals[0], np.ndarray):
            batched[key] = torch.from_numpy(np.stack(vals)).to(device)
        else:
            # Scalar-like (e.g. bool mask stored as np.bool_)
            batched[key] = torch.tensor(
                np.array(vals), device=device
            )
    return batched


def batch_infer(
    policy: _policy.Policy,
    obs_list: list[dict],
    batch_size: int = 50,
) -> list[dict]:
    """Run batched inference through the policy model.

    For **PyTorch** models: applies input transforms per-sample, stacks into a
    fixed-size batch, calls ``_sample_actions``, then applies output transforms.

    For **JAX** models: falls back to sequential ``policy.infer()`` calls
    (JAX JIT handles its own batching/caching internally).

    Args:
        policy: A loaded ``Policy`` object with transforms and model.
        obs_list: List of raw observation dicts (same format as ``policy.infer()`` input).
        batch_size: Fixed batch size (only meaningful for PyTorch models).

    Returns:
        List of output dicts (same format as ``policy.infer()`` output, minus timing info).
    """
    # ---- JAX fallback: sequential infer ----
    if not getattr(policy, "_is_pytorch_model", False):
        results = []
        for obs in obs_list:
            result = policy.infer(obs)
            result.pop("policy_timing", None)
            results.append(result)
        return results

    # ---- PyTorch batched path ----
    all_results: list[dict] = []
    device = policy._pytorch_device

    for start in range(0, len(obs_list), batch_size):
        chunk = obs_list[start: start + batch_size]
        actual_size = len(chunk)

        # Pad to fixed batch_size so torch.compile sees a consistent shape
        while len(chunk) < batch_size:
            chunk.append(chunk[0])

        # 1. Apply input transforms per-sample (CPU-bound)
        transformed = []
        for obs in chunk:
            inp = jax.tree.map(lambda x: x, obs)  # shallow copy
            inp = policy._input_transform(inp)
            transformed.append(inp)

        # 2. Stack into batched tensors → GPU
        batched = _collate_transformed(transformed, device)

        # 3. Create Observation and run model forward pass
        observation = _model.Observation.from_dict(batched)
        actions = policy._sample_actions(device, observation)  # (B, horizon, dim)

        # 4. Split batch, apply output transforms per-sample, discard padding
        for j in range(actual_size):
            outputs = {
                "state": batched["state"][j].detach().cpu().numpy(),
                "actions": actions[j].detach().cpu().numpy(),
            }
            outputs = policy._output_transform(outputs)
            all_results.append(outputs)

    return all_results


def find_max_batch_size(
    policy: _policy.Policy,
    sample_obs: dict,
    start: int = 50,
    max_bs: int = 200,
) -> int:
    """Binary-search for the largest batch size that fits in GPU memory.

    Only meaningful for PyTorch models; returns ``start`` for JAX models.
    """
    if not getattr(policy, "_is_pytorch_model", False):
        return start  # JAX handles its own batching
    best = 1
    bs = start
    while bs <= max_bs:
        try:
            torch.cuda.empty_cache()
            batch_infer(policy, [sample_obs] * bs, batch_size=bs)
            torch.cuda.synchronize()
            best = bs
            logger.info(f"  batch_size={bs} OK")
            bs = int(bs * 1.5)
        except torch.cuda.OutOfMemoryError:
            logger.info(f"  batch_size={bs} OOM — stopping")
            torch.cuda.empty_cache()
            break
    return best


# ============================== Model Loading =============================


def load_policy(cfg: EMAPConfig) -> _policy.Policy:
    """Download the fine-tuned checkpoint and build a ``Policy`` for local inference."""
    from huggingface_hub import snapshot_download, logging as hf_logging
    from openpi.shared import normalize as _normalize

    # Enable verbose HuggingFace Hub logging so download progress is visible
    hf_logging.set_verbosity_debug()

    logger.info(f"Downloading checkpoint from {cfg.checkpoint_repo} ...")
    checkpoint_dir = snapshot_download(cfg.checkpoint_repo)
    logger.info(f"Checkpoint at: {checkpoint_dir}")

    # Restore default HF logging level after download
    hf_logging.set_verbosity_warning()

    # Auto-discover norm_stats: the checkpoint may store them under a non-standard
    # asset_id (e.g. "new_all_lerobot_with_long/VLA_Arena" instead of
    # "datasets/vla-arena-lerobot"). Search for norm_stats.json within assets/.
    norm_stats = None
    assets_dir = pathlib.Path(checkpoint_dir) / "assets"
    if assets_dir.exists():
        for ns_path in assets_dir.rglob("norm_stats.json"):
            logger.info(f"Found norm_stats at: {ns_path}")
            norm_stats = _normalize.load(ns_path.parent)
            break
    if norm_stats is None:
        logger.warning("No norm_stats.json found in checkpoint assets — "
                       "normalization will rely on the data config defaults.")

    train_config = _config.get_config(cfg.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device=cfg.device,
        norm_stats=norm_stats,
    )
    logger.info(f"Policy loaded on {cfg.device}")
    return policy


# ============================= Data Loading ===============================


class ParquetDataset:
    """Lightweight random-access dataset backed by a directory of parquet files.

    This bypasses ``LeRobotDataset``'s ``datasets`` dependency (which may have
    version conflicts with the installed ``datasets`` library) and reads rows
    directly via PyArrow.
    """

    def __init__(self, data_dir: str | pathlib.Path):
        import pyarrow.parquet as pq

        data_dir = pathlib.Path(data_dir)
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found under {data_dir}")
        logger.info(f"Loading {len(parquet_files)} parquet files from {data_dir} ...")
        self._table = pq.read_table(parquet_files)
        logger.info(f"Loaded {len(self._table)} rows")

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx: int) -> dict:
        """Return a single row as a plain Python dict."""
        row = {col: self._table.column(col)[idx].as_py() for col in self._table.column_names}
        # Decode image bytes into numpy arrays
        for img_key in ("image", "wrist_image"):
            if img_key in row and isinstance(row[img_key], dict) and "bytes" in row[img_key]:
                import io
                from PIL import Image as PILImage
                row[img_key] = np.asarray(PILImage.open(io.BytesIO(row[img_key]["bytes"])))
        return row


def load_dataset(cfg: EMAPConfig):
    """Load the LeRobot dataset and return (dataset, metadata).

    Returns:
        dataset: ``ParquetDataset`` with single-frame random access.
        meta: ``LeRobotDatasetMetadata`` containing episode boundaries and tasks.
    """
    logger.info(f"Loading dataset metadata from {cfg.dataset_repo_id} ...")
    # Use revision="main" to bypass the lerobot CODEBASE_VERSION tag check –
    # the VLA-Arena dataset may not carry a v2.1 tag on the Hub.
    #
    # The VLA-Arena dataset nests all data under a VLA_Arena/ subdirectory.
    # We must point `root` to that subdirectory so lerobot finds meta/info.json.
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    default_root = HF_LEROBOT_HOME / cfg.dataset_repo_id
    nested_root = default_root / "VLA_Arena"
    # Use the nested root if it contains metadata, otherwise fall back to default
    root = str(nested_root) if (nested_root / "meta" / "info.json").exists() else None
    meta = lerobot_dataset.LeRobotDatasetMetadata(cfg.dataset_repo_id, root=root, revision="main")

    # Load parquet data directly (bypasses datasets library version issues)
    data_root = pathlib.Path(root) if root else default_root
    dataset = ParquetDataset(data_root / "data")
    num_episodes = meta.total_episodes
    logger.info(f"Dataset loaded: {len(dataset)} frames across {num_episodes} episodes")
    return dataset, meta


def build_observation(dataset, row_idx: int, prompt: str) -> dict:
    """Build an observation dict from a dataset row, ready for ``policy.infer()`` / ``batch_infer()``.

    Args:
        dataset: ``ParquetDataset`` or ``LeRobotDataset`` instance.
        row_idx: Global row index into the dataset.
        prompt: Language instruction string.

    Returns:
        Observation dict with keys expected by ``LiberoInputs``.
    """
    sample = dataset[row_idx]
    # Raw parquet uses "image"/"wrist_image"/"state"; map to observation/* keys
    # expected by LiberoInputs.
    obs = {
        "observation/image": np.asarray(sample["image"]),
        "observation/wrist_image": np.asarray(sample["wrist_image"]),
        "observation/state": np.asarray(sample["state"], dtype=np.float32),
        "prompt": prompt,
    }
    return obs


def extract_delta_actions(
    result: dict,
    obs: dict,
    eval_horizon: int = 10,
) -> np.ndarray:
    """Convert absolute-pose policy output back to delta actions.

    Args:
        result: Output dict from ``policy.infer()`` or ``batch_infer()``.
        obs: The input observation dict (used for ``observation/state[:6]``).
        eval_horizon: Number of action steps to keep.

    Returns:
        Flat delta-action vector of shape ``(eval_horizon * 7,)``.
    """
    actions = result["actions"][:eval_horizon, :]  # (H, 7) absolute EEF poses
    delta = actions.copy()
    # Subtract current EEF pose from dims 0-5 (position + orientation)
    # Dim 6 (gripper) stays absolute — matches training DeltaActions(mask=(T,T,T,T,T,T,F))
    current_state = np.asarray(obs["observation/state"], dtype=np.float32)
    delta[:, :6] -= current_state[:6]
    return delta.reshape(-1)  # (H * 7,)


# ======================== Progress-Stratified Sampling =====================


def _get_episode_boundaries(meta) -> tuple[list[int], list[int]]:
    """Compute (ep_from, ep_to) arrays from ``meta.episodes``.

    ``ep_from[i]`` is the global row index of the first frame of episode i.
    ``ep_to[i]`` is one past the last frame (exclusive).
    """
    ep_from: list[int] = []
    ep_to: list[int] = []
    cumulative = 0
    # meta.episodes is a dict {ep_idx: {..., "length": N}}; iterate in order
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
    """Compute the progress-stratified sampling plan.

    For each episode, compute frame indices at ``num_buckets`` evenly-spaced relative
    positions within the valid range ``[margin, N_e - margin]``.  Episodes shorter than
    ``2 * margin + 1`` frames are skipped.

    Args:
        meta: ``LeRobotDatasetMetadata`` with ``episodes`` list.
        num_buckets: Number of progress buckets (default 5 → 0%, 25%, 50%, 75%, 100%).
        margin: Frame margin from each end (= eval_horizon).

    Returns:
        Dict with:
            - ``bucket_samples``: list of ``num_buckets`` lists, each containing
              ``(episode_idx, global_row_idx)`` tuples.
            - ``skipped_episodes``: list of episode indices that were too short.
    """
    ep_from, ep_to = _get_episode_boundaries(meta)
    num_episodes = len(ep_from)

    bucket_fractions = np.linspace(0.0, 1.0, num_buckets)
    bucket_samples: list[list[tuple[int, int]]] = [[] for _ in range(num_buckets)]
    skipped_episodes: list[int] = []

    for ep_idx in range(num_episodes):
        start = int(ep_from[ep_idx])
        end = int(ep_to[ep_idx])
        ep_len = end - start

        if ep_len < 2 * margin + 1:
            skipped_episodes.append(ep_idx)
            continue

        valid_start = start + margin
        valid_end = end - margin  # exclusive
        valid_len = valid_end - valid_start  # >= 1

        for b_idx, frac in enumerate(bucket_fractions):
            # Map fraction to a frame within [valid_start, valid_end - 1]
            frame_offset = int(round(frac * (valid_len - 1)))
            global_row = valid_start + frame_offset
            bucket_samples[b_idx].append((ep_idx, global_row))

    return {
        "bucket_samples": bucket_samples,
        "skipped_episodes": skipped_episodes,
    }


def build_task_map(meta) -> dict[int, str]:
    """Return mapping from episode index to language instruction.

    Uses ``meta.episodes`` which stores ``{'tasks': ['instruction']}`` per episode.
    """
    return {
        ep_idx: ep_data["tasks"][0]
        for ep_idx, ep_data in meta.episodes.items()
    }


# ============================ Inference Loop ==============================


def collect_marginals(
    cfg: EMAPConfig,
    policy: _policy.Policy,
    dataset,
    meta,
) -> dict:
    """Main EMAP inference loop.

    For each progress bucket and each anchor sample:
      1. Compute ``f_full`` — the true delta-action prediction.
      2. Compute ``vis_marginal`` — mean delta-action over K random instructions (E_L).
      3. Compute ``lang_marginal`` — mean delta-action over M random (image, state) from same bucket (E_V).

    Results are checkpointed to disk after each bucket.

    Returns:
        Dict with keys ``f_full``, ``vis_marginals``, ``lang_marginals``, each a list of
        1-D numpy arrays (flattened 70D delta actions).
    """
    rng = np.random.default_rng(cfg.seed)
    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------- Sampling plan -------
    plan = compute_sample_plan(meta, cfg.num_buckets, margin=cfg.eval_horizon)
    bucket_samples = plan["bucket_samples"]
    logger.info(
        f"Sampling plan: {cfg.num_buckets} buckets, "
        f"{sum(len(b) for b in bucket_samples)} total samples, "
        f"{len(plan['skipped_episodes'])} episodes skipped"
    )

    # ------- Task (instruction) map -------
    ep_instructions = build_task_map(meta)  # {ep_idx: instruction_str}
    ep_from, ep_to = _get_episode_boundaries(meta)
    num_episodes = len(ep_from)

    all_instructions = list(set(ep_instructions.values()))
    logger.info(f"Unique instructions: {len(all_instructions)}")

    # ------- Auto batch size -------
    actual_batch_size = cfg.batch_size
    if cfg.auto_batch:
        logger.info("Auto-tuning batch size ...")
        # Use first anchor as a test sample
        first_ep, first_row = bucket_samples[0][0]
        sample_obs = build_observation(dataset, first_row, ep_instructions[first_ep])
        actual_batch_size = find_max_batch_size(policy, sample_obs, start=cfg.batch_size)
        logger.info(f"Auto-tuned batch_size = {actual_batch_size}")

    # ------- Collect results -------
    all_f_full: list[np.ndarray] = []
    all_vis_marginals: list[np.ndarray] = []
    all_lang_marginals: list[np.ndarray] = []

    total_inferences = 0
    wall_start = time.time()

    for b_idx in range(cfg.num_buckets):
        bucket = bucket_samples[b_idx]
        bucket_tag = f"bucket_{b_idx}"
        checkpoint_path = output_dir / f"checkpoint_{bucket_tag}.pt"

        # Resume from checkpoint if available
        if checkpoint_path.exists():
            logger.info(f"Resuming {bucket_tag} from checkpoint")
            ckpt = torch.load(checkpoint_path, weights_only=False)
            all_f_full.extend([np.array(x) for x in ckpt["f_full"]])
            all_vis_marginals.extend([np.array(x) for x in ckpt["vis_marginals"]])
            all_lang_marginals.extend([np.array(x) for x in ckpt["lang_marginals"]])
            total_inferences += ckpt.get("num_inferences", 0)
            continue

        bucket_f_full: list[np.ndarray] = []
        bucket_vis_marginals: list[np.ndarray] = []
        bucket_lang_marginals: list[np.ndarray] = []
        bucket_inferences = 0

        # Pre-collect the (image, state) donor pool for language marginals:
        # All samples in this bucket from OTHER episodes
        donor_pool = bucket  # list of (ep_idx, global_row_idx)

        logger.info(
            f"[{bucket_tag}] Processing {len(bucket)} anchor samples (K={cfg.K}, M={cfg.M}) ..."
        )

        for anchor_idx, (ep_idx, row_idx) in enumerate(
            tqdm.tqdm(bucket, desc=bucket_tag, leave=True)
        ):
            instruction = ep_instructions[ep_idx]
            anchor_obs = build_observation(dataset, row_idx, instruction)

            # ---- 1. True action (f_full) — collect all at once later? ----
            # We batch true actions together with other anchors below.
            # For simplicity, compute one at a time within the marginal batches.

            # ---- 2. Visual marginals: same (image, state), K random instructions ----
            sampled_instructions = rng.choice(
                all_instructions, size=cfg.K, replace=True
            )
            vis_obs_list = []
            for instr in sampled_instructions:
                vis_obs_list.append({
                    "observation/image": anchor_obs["observation/image"],
                    "observation/wrist_image": anchor_obs["observation/wrist_image"],
                    "observation/state": anchor_obs["observation/state"].copy(),
                    "prompt": instr,
                })

            vis_results = batch_infer(policy, vis_obs_list, batch_size=actual_batch_size)
            vis_deltas = np.stack([
                extract_delta_actions(r, anchor_obs, cfg.eval_horizon)
                for r in vis_results
            ])  # (K, 70)
            vis_marginal = vis_deltas.mean(axis=0)  # (70,)

            # ---- 3. Language marginals: same instruction, M random (image, state) from same bucket ----
            # Exclude self from donor pool
            eligible = [(e, r) for (e, r) in donor_pool if e != ep_idx]
            if len(eligible) < cfg.M:
                # If not enough other episodes, allow sampling with replacement
                donor_indices = rng.choice(len(eligible), size=cfg.M, replace=True)
            else:
                donor_indices = rng.choice(len(eligible), size=cfg.M, replace=False)

            lang_obs_list = []
            lang_donor_obs_list = []  # keep track for delta extraction
            for d_idx in donor_indices:
                d_ep, d_row = eligible[d_idx]
                donor_obs = build_observation(dataset, d_row, instruction)
                lang_obs_list.append(donor_obs)
                lang_donor_obs_list.append(donor_obs)

            lang_results = batch_infer(policy, lang_obs_list, batch_size=actual_batch_size)
            lang_deltas = np.stack([
                extract_delta_actions(r, lang_donor_obs_list[i], cfg.eval_horizon)
                for i, r in enumerate(lang_results)
            ])  # (M, 70)
            lang_marginal = lang_deltas.mean(axis=0)  # (70,)

            # ---- 4. True action (f_full) from anchor observation ----
            true_results = batch_infer(policy, [anchor_obs], batch_size=actual_batch_size)
            f_full = extract_delta_actions(true_results[0], anchor_obs, cfg.eval_horizon)

            bucket_f_full.append(f_full)
            bucket_vis_marginals.append(vis_marginal)
            bucket_lang_marginals.append(lang_marginal)
            bucket_inferences += 1 + cfg.K + cfg.M

        # Checkpoint this bucket
        torch.save(
            {
                "f_full": bucket_f_full,
                "vis_marginals": bucket_vis_marginals,
                "lang_marginals": bucket_lang_marginals,
                "num_inferences": bucket_inferences,
                "bucket_idx": b_idx,
                "num_samples": len(bucket),
            },
            checkpoint_path,
        )
        logger.info(
            f"[{bucket_tag}] Saved checkpoint ({len(bucket)} samples, "
            f"{bucket_inferences} inferences)"
        )

        all_f_full.extend(bucket_f_full)
        all_vis_marginals.extend(bucket_vis_marginals)
        all_lang_marginals.extend(bucket_lang_marginals)
        total_inferences += bucket_inferences

    wall_elapsed = time.time() - wall_start
    logger.info(
        f"Collection complete: {total_inferences} inferences in {wall_elapsed:.1f}s "
        f"({total_inferences / max(wall_elapsed, 1e-9):.1f} inf/s)"
    )

    return {
        "f_full": all_f_full,
        "vis_marginals": all_vis_marginals,
        "lang_marginals": all_lang_marginals,
    }


# =============================== Metrics =================================


def compute_emap_metrics(
    results: dict,
    cfg: EMAPConfig,
) -> dict:
    """Compute EMAP metrics from collected marginal results.

    Steps:
      1. Per-dimension z-score normalization of all 70D delta-action vectors.
      2. Compute global mean mu.
      3. Compute f_add = vis_marginal + lang_marginal - mu.
      4. Compute Interaction Ratio and R-squared.

    Args:
        results: Dict with ``f_full``, ``vis_marginals``, ``lang_marginals``
                 (each a list of 1-D numpy arrays).
        cfg: EMAP configuration.

    Returns:
        Dict with computed metrics and metadata.
    """
    f_full = np.stack(results["f_full"])  # (N, 70)
    vis_marginals = np.stack(results["vis_marginals"])  # (N, 70)
    lang_marginals = np.stack(results["lang_marginals"])  # (N, 70)
    N = f_full.shape[0]
    D = f_full.shape[1]
    logger.info(f"Computing EMAP metrics: N={N} samples, D={D} dims")

    # ---- 1. Per-dimension z-score normalization ----
    mu_norm = f_full.mean(axis=0)  # (70,)
    std_norm = f_full.std(axis=0) + 1e-8  # (70,)

    f_full_z = (f_full - mu_norm) / std_norm
    vis_z = (vis_marginals - mu_norm) / std_norm
    lang_z = (lang_marginals - mu_norm) / std_norm

    # ---- 2. Global mean (post-normalization) ----
    mu = f_full_z.mean(axis=0)  # (70,)  ≈ 0 after z-score, but compute exactly

    # ---- 3. Additive projection ----
    f_add = vis_z + lang_z - mu  # (N, 70)

    # ---- 4. Metrics ----
    # Residuals
    ss_interaction = np.sum((f_full_z - f_add) ** 2)
    ss_total = np.sum((f_full_z - mu) ** 2)

    interaction_ratio = float(ss_interaction / (ss_total + 1e-12))
    r_squared = 1.0 - interaction_ratio

    logger.info(f"EMAP R^2 = {r_squared:.6f}")
    logger.info(f"EMAP Interaction Ratio = {interaction_ratio:.6f}")

    # ---- 5. Per-bucket breakdown ----
    bucket_size = N // cfg.num_buckets
    per_bucket: list[dict] = []
    for b in range(cfg.num_buckets):
        s = b * bucket_size
        e = s + bucket_size if b < cfg.num_buckets - 1 else N
        b_f = f_full_z[s:e]
        b_add = f_add[s:e]
        b_mu = mu  # same global mu
        b_ss_int = np.sum((b_f - b_add) ** 2)
        b_ss_tot = np.sum((b_f - b_mu) ** 2)
        b_ir = float(b_ss_int / (b_ss_tot + 1e-12))
        per_bucket.append({
            "bucket": b,
            "progress_pct": f"{b / max(cfg.num_buckets - 1, 1) * 100:.0f}%",
            "num_samples": e - s,
            "interaction_ratio": b_ir,
            "r_squared": 1.0 - b_ir,
        })

    # ---- 6. Per-dimension R^2 ----
    per_dim_r2: list[float] = []
    for d in range(D):
        ss_int_d = np.sum((f_full_z[:, d] - f_add[:, d]) ** 2)
        ss_tot_d = np.sum((f_full_z[:, d] - mu[d]) ** 2)
        per_dim_r2.append(float(1.0 - ss_int_d / (ss_tot_d + 1e-12)))

    summary = {
        "r_squared": r_squared,
        "interaction_ratio": interaction_ratio,
        "num_samples": N,
        "action_dim": D,
        "eval_horizon": cfg.eval_horizon,
        "K": cfg.K,
        "M": cfg.M,
        "num_buckets": cfg.num_buckets,
        "per_bucket": per_bucket,
        "per_dim_r2": per_dim_r2,
        "normalization": {
            "mean": mu_norm.tolist(),
            "std": std_norm.tolist(),
        },
    }

    return summary


# ================================= Main ===================================


def main(cfg: EMAPConfig) -> None:
    """EMAP evaluator entry point."""
    logger.info("=" * 60)
    logger.info("EMAP Evaluator for Pi-Zero on VLA-Arena")
    logger.info("=" * 60)
    logger.info(f"Config: {cfg}")

    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "emap_config.json"
    config_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    logger.info(f"Config saved to {config_path}")

    # 1. Load model
    logger.info("----- Loading model -----")
    policy = load_policy(cfg)

    # 2. Load dataset
    logger.info("----- Loading dataset -----")
    dataset, meta = load_dataset(cfg)

    # 3. Collect marginals (main inference loop)
    logger.info("----- Collecting marginals -----")
    results = collect_marginals(cfg, policy, dataset, meta)

    # 4. Compute metrics
    logger.info("----- Computing metrics -----")
    summary = compute_emap_metrics(results, cfg)

    # 5. Save results
    results_path = output_dir / "emap_results.json"
    results_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Results saved to {results_path}")

    # Also save the raw vectors for further analysis
    raw_path = output_dir / "emap_raw_vectors.pt"
    torch.save(
        {
            "f_full": np.stack(results["f_full"]),
            "vis_marginals": np.stack(results["vis_marginals"]),
            "lang_marginals": np.stack(results["lang_marginals"]),
        },
        raw_path,
    )
    logger.info(f"Raw vectors saved to {raw_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info(f"EMAP R^2            = {summary['r_squared']:.6f}")
    logger.info(f"Interaction Ratio   = {summary['interaction_ratio']:.6f}")
    logger.info(f"Num samples         = {summary['num_samples']}")
    logger.info("Per-bucket breakdown:")
    for b in summary["per_bucket"]:
        logger.info(
            f"  Bucket {b['bucket']} ({b['progress_pct']}): "
            f"R^2={b['r_squared']:.4f}, IR={b['interaction_ratio']:.4f}, "
            f"N={b['num_samples']}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
