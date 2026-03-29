"""Extract attention weights, Q-projections, t-SNE, and neighbors from Pi-Zero.

End-to-end pipeline:
  1. Load checkpoint + transform chain via ``create_trained_policy``.
  2. Stream episodes from the VLA-Arena HuggingFace dataset.
  3. For each sampled timestep, run ``sample_actions`` (JIT) to get denoised x_0,
     then run a capture pass (non-JIT) to record attention + Q-projections.
  4. Pre-compute t-SNE and nearest neighbors.
  5. Serialize everything to HDF5.

Usage::

    cd <openpi_root>
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform HF_HUB_OFFLINE=1 \\
      .venv/bin/python src/openpi_interpret/extraction/extract_interpret_data.py \\
        [--checkpoint /path/to/checkpoint] \\
        --max-episodes 3 --timestep-stride 40

The script sets ``HF_HUB_OFFLINE=1`` so Hugging Face Hub uses only the local cache
(no accidental downloads). Ensure the Pi-Zero checkpoint and LeRobot dataset
are cached first. The default ``--checkpoint`` is the known-good hub snapshot
under ``~/.cache/huggingface/hub/.../acdc8e7eaa6dfccedef6db26626ec828bfa21b1e``;
newer snapshots (e.g. ``9fb1694...``) may ship a corrupted OCDBT blob.
"""

from __future__ import annotations

import argparse
import io
import os
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

OPENPI_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(OPENPI_ROOT / "src"))

from openpi_interpret.extraction import capture, serialize, tsne  # noqa: E402

logger = logging.getLogger(__name__)

# Default: working HF hub snapshot (avoid snapshot_download picking a broken revision).
_DEFAULT_PI0_CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub/models--VLA-Arena--pi0-vla-arena-fintuned/snapshots/acdc8e7eaa6dfccedef6db26626ec828bfa21b1e"
)

SAMPLED_LAYERS: list[int] = [0, 3, 6, 9, 12, 15, 17]
CAMERA_NAMES: list[str] = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
IMAGE_KEYS_MAP: dict[str, str] = {
    "observation.images.image": "base_0_rgb",
    "observation.images.wrist_image": "left_wrist_0_rgb",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract Pi-Zero interpretability data.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_DEFAULT_PI0_CHECKPOINT),
        help=(
            "Path to model checkpoint directory (local path; not snapshot_download). "
            f"Default: known-good hub snapshot {_DEFAULT_PI0_CHECKPOINT}"
        ),
    )
    parser.add_argument("--dataset-repo-id", type=str, default="VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data"))
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--timestep-stride", type=int, default=30)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--config-name", type=str, default="pi0_vla_arena_low_mem_finetune")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_model_and_policy(config_name: str, checkpoint_dir: Path | str):
    """Load the Pi-Zero model and policy transform chain.

    Returns:
        Tuple of (policy, model, input_transform).
    """
    import jax

    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    logger.info("Loading config %r and checkpoint from %s", config_name, checkpoint_dir)
    config = get_config(config_name)
    policy = create_trained_policy(config, Path(checkpoint_dir).expanduser().resolve())
    model = policy._model
    logger.info("Model loaded on %s", jax.devices())
    return policy, model


def load_episodes(
    repo_id: str,
    max_episodes: int,
) -> list[dict]:
    """Download episode metadata and parquet data from HuggingFace.

    Returns:
        List of episode dicts with keys: episode_index, task, length, rows.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    logger.info("Loading dataset metadata from %s", repo_id)
    meta_path = hf_hub_download(
        repo_id, "meta/episodes.jsonl", repo_type="dataset", local_files_only=True,
    )
    tasks_path = hf_hub_download(
        repo_id, "meta/tasks.jsonl", repo_type="dataset", local_files_only=True,
    )

    with open(tasks_path) as f:
        tasks = {row["task_index"]: row["task"] for row in (json.loads(l) for l in f)}
    with open(meta_path) as f:
        episodes = [json.loads(l) for l in f]

    selected = episodes[:max_episodes]
    result = []
    for ep in selected:
        ep_idx = ep["episode_index"]
        fname = f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        try:
            local = hf_hub_download(
                repo_id, fname, repo_type="dataset", local_files_only=True,
            )
        except Exception:
            logger.warning("Parquet %s not in local cache, skipping episode %d", fname, ep_idx)
            continue
        df = pd.read_parquet(local)
        task_str = _resolve_task(df, ep, tasks)
        result.append({
            "episode_index": ep_idx,
            "task": task_str,
            "length": len(df),
            "rows": df,
        })
    logger.info("Loaded %d episodes", len(result))
    return result


def _resolve_task(df, ep_meta: dict, tasks: dict[int, str]) -> str:
    """Extract the task instruction string for an episode."""
    if "tasks" in ep_meta and ep_meta["tasks"]:
        return ep_meta["tasks"][0]
    task_idx = int(df["task_index"].iloc[0])
    return tasks.get(task_idx, f"task_{task_idx}")


def build_observation(row, policy) -> dict:
    """Convert a single dataset row into a policy-ready observation dict.

    Applies the policy's input transforms and returns a dict ready for
    ``Observation.from_dict``.
    """
    import jax.numpy as jnp
    import numpy as np

    base_img = _decode_image(row["observation.images.image"])
    wrist_img = _decode_image(row["observation.images.wrist_image"])
    state = np.array(row["observation.state"], dtype=np.float32)

    obs_dict = {
        "observation/image": base_img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": row.get("prompt", row.get("_task", "")),
    }

    import jax

    transformed = policy._input_transform(obs_dict)
    batched = jax.tree.map(
        lambda x: jnp.asarray(x)[np.newaxis, ...], transformed
    )
    return batched


def _decode_image(img_field) -> np.ndarray:
    """Decode an image from the dataset (bytes dict or PIL) to uint8 [H, W, 3]."""
    if isinstance(img_field, dict) and "bytes" in img_field:
        pil_img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    if isinstance(img_field, Image.Image):
        return np.array(img_field.convert("RGB"), dtype=np.uint8)
    return np.array(img_field, dtype=np.uint8)


def build_token_meta(prefix_len: int, suffix_len: int, instruction_tokens: list[str]) -> list[dict]:
    """Build the 867-entry token metadata list.

    Token layout: 3×256 image patches + up to 48 language tokens + 1 state + 50 actions.
    """
    meta: list[dict] = []

    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        for patch_idx in range(256):
            row = patch_idx // 16
            col = patch_idx % 16
            meta.append({
                "index": len(meta),
                "type": "image_patch",
                "source": cam_name,
                "patch_row": row,
                "patch_col": col,
                "token_text": None,
                "token_position": None,
            })

    num_lang_tokens = prefix_len - 768
    for i in range(num_lang_tokens):
        tok_text = instruction_tokens[i] if i < len(instruction_tokens) else "[PAD]"
        meta.append({
            "index": len(meta),
            "type": "language",
            "source": "language",
            "patch_row": None,
            "patch_col": None,
            "token_text": tok_text,
            "token_position": i,
        })

    meta.append({
        "index": len(meta),
        "type": "state",
        "source": "state",
        "patch_row": None,
        "patch_col": None,
        "token_text": None,
        "token_position": None,
    })

    for i in range(suffix_len - 1):
        meta.append({
            "index": len(meta),
            "type": "action",
            "source": "action",
            "patch_row": None,
            "patch_col": None,
            "token_text": None,
            "token_position": i,
        })

    return meta


def get_instruction_tokens(prompt: str) -> list[str]:
    """Tokenize a prompt and return individual token strings for metadata.

    Uses the PaliGemma sentencepiece tokenizer.
    """
    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer(max_len=48)
    tokens, mask = tokenizer.tokenize(prompt)
    num_real = int(mask.sum())
    pieces = []
    for tid in tokens[:num_real]:
        tid = int(tid)
        try:
            pieces.append(tokenizer._tokenizer.id_to_piece(tid))
        except Exception:
            pieces.append(f"[{tid}]")
    return pieces


def extract_camera_images(row) -> dict[str, np.ndarray]:
    """Extract camera images from a dataset row as uint8 [224, 224, 3]."""
    images: dict[str, np.ndarray] = {}
    base_img = _decode_image(row["observation.images.image"])
    wrist_img = _decode_image(row["observation.images.wrist_image"])
    base_img = _resize_if_needed(base_img, 224, 224)
    wrist_img = _resize_if_needed(wrist_img, 224, 224)
    images["base_0_rgb"] = base_img
    images["left_wrist_0_rgb"] = wrist_img
    images["right_wrist_0_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)
    return images


def _resize_if_needed(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize image to (h, w) if needed."""
    if img.shape[0] != h or img.shape[1] != w:
        pil = Image.fromarray(img).resize((w, h), Image.BILINEAR)
        return np.array(pil, dtype=np.uint8)
    return img


def process_episode(
    episode: dict,
    policy,
    model,
    output_dir: Path,
    timestep_stride: int,
) -> Path:
    """Run the full extraction pipeline for a single episode.

    Args:
        episode: Dict with episode_index, task, length, rows.
        policy: Loaded Policy instance.
        model: Pi0 model instance.
        output_dir: Directory for HDF5 output.
        timestep_stride: Sample every N-th frame.

    Returns:
        Path to the written HDF5 file.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation

    ep_idx = episode["episode_index"]
    task = episode["task"]
    df = episode["rows"]
    episode_id = f"ep_{ep_idx:06d}"

    logger.info("Processing episode %s (%d frames, task=%r)", episode_id, len(df), task)

    frame_indices = list(range(0, len(df), timestep_stride))
    if not frame_indices:
        frame_indices = [0]

    instruction_tokens = get_instruction_tokens(task)

    import gc
    import jax

    timestep_results: list[dict] = []
    for frame_idx in frame_indices:
        logger.info("  Frame %d/%d", frame_idx, len(df) - 1)
        row = df.iloc[frame_idx].to_dict()
        row["prompt"] = task
        ts_data = _process_timestep(row, policy, model, frame_idx, instruction_tokens)
        ts_data["camera_images"] = extract_camera_images(row)
        timestep_results.append(ts_data)
        gc.collect()
        jax.clear_caches()

    capture.uninstall_hooks()

    return serialize.write_episode_hdf5(
        output_dir=output_dir,
        episode_id=episode_id,
        task_instruction=task,
        instruction_tokens=instruction_tokens,
        timestep_data=timestep_results,
    )


def _process_timestep(
    row: dict,
    policy,
    model,
    frame_idx: int,
    instruction_tokens: list[str],
) -> dict:
    """Extract attention, Q-projections, t-SNE, and neighbors for one timestep."""
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation

    batched = build_observation(row, policy)
    observation = Observation.from_dict(batched)

    rng = jax.random.key(frame_idx)
    from openpi.models.model import preprocess_observation
    obs_processed = preprocess_observation(None, observation, train=False)

    logger.info("    Running sample_actions (JIT)...")
    t0 = time.time()
    x_0 = model.sample_actions(rng, obs_processed, num_steps=10)
    x_0.block_until_ready()
    logger.info("    sample_actions took %.1fs", time.time() - t0)

    logger.info("    Running capture pass...")
    t0 = time.time()
    captured = capture.run_capture_pass(model, obs_processed, x_0, timestep=0.1)
    logger.info("    Capture pass took %.1fs", time.time() - t0)

    token_meta = build_token_meta(captured.prefix_len, captured.suffix_len, instruction_tokens)

    tsne_results: dict[int, np.ndarray] = {}
    neighbor_results: dict[int, np.ndarray] = {}
    for layer_idx in SAMPLED_LAYERS:
        if layer_idx in captured.q_prefix and layer_idx in captured.q_suffix:
            logger.info("    t-SNE for layer %d...", layer_idx)
            coords, neighbors = tsne.compute_layer_tsne_and_neighbors(
                captured.q_prefix[layer_idx],
                captured.q_suffix[layer_idx],
            )
            tsne_results[layer_idx] = coords
            neighbor_results[layer_idx] = neighbors

    q_prefix_squeezed = {k: v[0] for k, v in captured.q_prefix.items()}
    q_suffix_squeezed = {k: v[0] for k, v in captured.q_suffix.items()}

    return {
        "timestep": frame_idx,
        "token_meta": token_meta,
        "attention": captured.attention,
        "tsne": tsne_results,
        "neighbors": neighbor_results,
        "q_prefix": q_prefix_squeezed,
        "q_suffix": q_suffix_squeezed,
    }


def main() -> None:
    """Entry point for the extraction pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Pi-Zero Interpretability Data Extraction ===")
    logger.info("Using local_files_only=True for all HF Hub calls (cache-only, no downloads)")
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {ckpt_path}. "
            "Pass --checkpoint to a local Pi-Zero checkpoint or cache the working snapshot."
        )
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Dataset: %s", args.dataset_repo_id)
    logger.info("Output: %s", output_dir)
    logger.info("Max episodes: %d, stride: %d", args.max_episodes, args.timestep_stride)

    policy, model = load_model_and_policy(args.config_name, ckpt_path)
    episodes = load_episodes(args.dataset_repo_id, args.max_episodes)

    import gc
    import jax

    written_paths: list[Path] = []
    for ep in episodes:
        path = process_episode(ep, policy, model, output_dir, args.timestep_stride)
        written_paths.append(path)
        gc.collect()
        jax.clear_caches()
        logger.info("Cleared JAX caches and ran GC between episodes")

    logger.info("=== Extraction Complete ===")
    for p in written_paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        logger.info("  %s: %.1f MB", p.name, size_mb)


if __name__ == "__main__":
    main()
