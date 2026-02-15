#!/usr/bin/env python3
# Copyright 2025 The CMU MMML Team 2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLA-Arena LeRobot OpenPi Visualizer Data Preprocessor

Converts LeRobot OpenPi datasets from "image" mode (PNG bytes embedded in
Parquet files) to "video" mode (standalone MP4 files) so that the
lerobot-dataset-visualizer can play them back in real time.

The original raw dataset is left untouched.  A new, visualizer-ready copy
is written to *--output-dir* (default: ``<dataset_dir>_viz``).

**Why a dedicated preprocessing step?**

The visualizer runs in the browser and needs to play video at 10 FPS.
Extracting individual PNG frames from Parquet at runtime requires a Python
subprocess per frame (~440 ms each), which is 4x too slow for smooth
playback.  By converting the dataset once up front, the visualizer can use
the browser's native ``<video>`` element for instant, perfectly-synced
playback.

Usage:
    # Process a downloaded dataset (auto-detects subset prefix)
    vla-arena.prepare-viz /path/to/VLA_Arena_L0_S_lerobot_openpi

    # Process with a custom output directory
    vla-arena.prepare-viz /path/to/dataset --output-dir /data/viz

    # Force re-processing of already-converted episodes
    vla-arena.prepare-viz /path/to/dataset --force

    # Keep original image columns in parquet (larger but lossless)
    vla-arena.prepare-viz /path/to/dataset --keep-images

    # Use 4 parallel workers for faster processing
    vla-arena.prepare-viz /path/to/dataset --workers 4
"""

import argparse
import copy
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Ensure heavy dependencies are importable (may live in openpi's .venv)
# ---------------------------------------------------------------------------
_DEPS_READY = False


def _ensure_deps():
    """Make pyarrow, imageio and PIL importable.

    Checks the standard Python path first, then falls back to the
    site-packages inside openpi's uv-managed .venv.
    """
    global _DEPS_READY
    if _DEPS_READY:
        return

    try:
        import imageio  # noqa: F401
        import PIL  # noqa: F401
        import pyarrow  # noqa: F401

        _DEPS_READY = True
        return
    except ImportError:
        pass

    import glob as _glob

    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_pattern = os.path.join(
        script_dir,
        '..',
        'vla_arena',
        'models',
        'openpi',
        '.venv',
        'lib',
        'python*',
        'site-packages',
    )
    for site_dir in _glob.glob(venv_pattern):
        site_dir = os.path.normpath(site_dir)
        if site_dir not in sys.path:
            sys.path.insert(0, site_dir)

    try:
        import imageio  # noqa: F401
        import PIL  # noqa: F401
        import pyarrow  # noqa: F401

        _DEPS_READY = True
    except ImportError as exc:
        print(
            f'Error: Missing dependency {exc.name}.\n'
            'Install the required packages:\n'
            '  pip install pyarrow Pillow "imageio[ffmpeg]"\n'
            'or ensure the openpi venv is set up:\n'
            '  cd vla_arena/models/openpi && uv sync',
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Colour helper (graceful degradation when termcolor is absent)
# ---------------------------------------------------------------------------
try:
    from termcolor import colored
except ImportError:

    def colored(text, color=None, **kwargs):
        return text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H264_CODEC = 'libx264'
H264_PIX_FMT = 'yuv420p'
MP4_QUALITY = 5  # imageio quality (lower = better, 5 is high quality)

# Default video_info block mirroring the LeRobot v2.1 convention
DEFAULT_VIDEO_INFO = {
    'video.fps': 10,         # will be overridden by actual fps
    'video.codec': 'h264',   # H.264 — universal browser support
    'video.pix_fmt': H264_PIX_FMT,
    'video.is_depth_map': False,
    'has_audio': False,
}

# ---------------------------------------------------------------------------
# Per-dimension motor names for the Panda EEF state / action features.
#
# VLA-Arena records Franka Panda state / action in *end-effector* space,
# **not** joint space.  The full pipeline is:
#
#   create_dataset.py
#   ├── ee_states = eef_pos(3) + quat2axisangle(eef_quat)(3)  →  6D
#   └── gripper_states = obs['robot0_gripper_qpos']            →  2D
#         ↓
#   RLDS builder (VLA_Arena_dataset_builder.py)
#   ├── state(8D)  = concat(ee_states, gripper_states)
#   └── action(7D) = raw OSC_POSE controller input
#         ↓
#   convert_data_to_lerobot_openpi.py  (passthrough to LeRobot parquet)
#
# Orientation: ``quat2axisangle`` returns an *axis-angle* 3-vector whose
# direction encodes the rotation axis and whose magnitude is the rotation
# angle.  Although the SmolVLA conversion script labels these "roll",
# "pitch", "yaw", the underlying representation is axis-angle (confirmed
# by the data: orientation norms cluster tightly near π).
#
# Dimension mismatch (8 vs 7):
#   state  has 2D gripper (left + right finger *positions*)
#   action has 1D gripper (single open/close *command*;
#                          the OSC controller maps it to both fingers)
#
# The first 6 names are shared so the visualizer can group matching
# state–action dimensions into side-by-side charts.
#
# Source: scripts/create_dataset.py              (lines 213-220)
#         scripts/convert_data_to_lerobot_smolvla.py (lines 71-101)
#         vla_arena/models/openpi/evaluator.py   (lines 234-238)
#         rlds_dataset_builder/VLA_Arena/VLA_Arena_dataset_builder.py
# ---------------------------------------------------------------------------
PANDA_EEF_STATE_NAMES = {
    'motors': [
        'x', 'y', 'z',                       # EEF position (m)
        'roll', 'pitch', 'yaw',              # EEF axis-angle orientation (rad)
        'gripper_left', 'gripper_right',     # finger joint positions
    ],
}
PANDA_EEF_ACTION_NAMES = {
    'motors': [
        'x', 'y', 'z',                       # EEF position delta (normalized)
        'roll', 'pitch', 'yaw',              # EEF orient. delta  (normalized)
        'gripper',                           # gripper command
    ],
}


# ===================================================================
# Core: locate dataset root (handles subset prefix)
# ===================================================================
def find_dataset_root(dataset_dir: str) -> tuple[str, str | None]:
    """Return (meta_root, subset_prefix | None).

    VLA-Arena datasets live under a subset prefix like ``VLA_Arena/``.
    This function detects it automatically.
    """
    dataset_dir = os.path.normpath(dataset_dir)

    # Direct: info.json is at dataset_dir/meta/info.json
    if os.path.isfile(os.path.join(dataset_dir, 'meta', 'info.json')):
        return dataset_dir, None

    # Subset prefix: info.json is at dataset_dir/<prefix>/meta/info.json
    for entry in os.listdir(dataset_dir):
        subdir = os.path.join(dataset_dir, entry)
        if os.path.isdir(subdir) and not entry.startswith('.'):
            if os.path.isfile(os.path.join(subdir, 'meta', 'info.json')):
                return subdir, entry

    raise FileNotFoundError(
        f'Could not find meta/info.json inside {dataset_dir}.\n'
        'Is this a valid LeRobot dataset directory?'
    )


# ===================================================================
# Core: extract image columns from one parquet file
# ===================================================================
def _extract_image_columns(parquet_path: str, image_columns: list[str]):
    """Read *parquet_path* and return a dict mapping camera name to a list
    of PIL Images (one per row / time-step).
    """
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet_path, columns=image_columns)
    result: dict[str, list] = {col: [] for col in image_columns}
    for col_name in image_columns:
        column = table.column(col_name)
        for i in range(len(column)):
            cell = column[i].as_py()
            img_bytes = cell['bytes'] if isinstance(cell, dict) else cell
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            result[col_name].append(img)
    return result


# ===================================================================
# Core: encode a list of PIL Images → MP4 video
# ===================================================================
def _encode_video(
    frames: list,
    output_path: str,
    fps: int,
) -> None:
    """Encode *frames* (list of PIL Images) to an H.264 MP4 file."""
    import imageio
    import numpy as np

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec=H264_CODEC,
        pixelformat=H264_PIX_FMT,
        quality=MP4_QUALITY,
        # Place the moov atom before mdat so the browser can start
        # playing immediately without needing HTTP Range requests.
        output_params=['-movflags', '+faststart'],
    )
    for frame in frames:
        writer.append_data(np.asarray(frame))
    writer.close()


# ===================================================================
# Core: process one episode (extractable for parallel execution)
# ===================================================================
def _process_one_episode(
    src_parquet: str,
    image_columns: list[str],
    video_path_template: str,
    output_root: str,
    episode_index: int,
    chunks_size: int,
    fps: int,
    force: bool,
) -> dict:
    """Process a single episode parquet → MP4 videos.

    Returns a dict of ``{camera: video_path}`` for reporting.
    """
    _ensure_deps()

    episode_chunk = episode_index // chunks_size
    videos_written: dict[str, str] = {}

    for camera in image_columns:
        # Build the video destination path from the template
        rel_video = video_path_template.format(
            episode_chunk=episode_chunk,
            video_key=camera,
            episode_index=episode_index,
        )
        abs_video = os.path.join(output_root, rel_video)

        # Skip if already exists (idempotent)
        if os.path.isfile(abs_video) and not force:
            videos_written[camera] = rel_video
            continue

        # Extract frames for this camera only
        camera_frames = _extract_image_columns(src_parquet, [camera])
        frames = camera_frames[camera]
        if not frames:
            continue

        # Encode
        _encode_video(frames, abs_video, fps)
        videos_written[camera] = rel_video

    return videos_written


# ===================================================================
# Core: strip image columns from parquet, copy to output
# ===================================================================
def _copy_parquet_without_images(
    src_path: str,
    dst_path: str,
    image_columns: list[str],
) -> None:
    """Copy parquet file with image columns dropped."""
    import pyarrow.parquet as pq

    table = pq.read_table(src_path)
    # Drop image columns
    for col in image_columns:
        if col in table.column_names:
            idx = table.column_names.index(col)
            table = table.remove_column(idx)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    pq.write_table(table, dst_path)


# ===================================================================
# Core: build the updated info.json
# ===================================================================
def _build_viz_info(
    original_info: dict,
    image_columns: list[str],
    fps: int,
    total_episodes: int,
) -> dict:
    """Return a new info dict with image features converted to video.

    Also patches ``state`` / ``actions`` feature names with per-dimension
    motor labels so the visualizer renders individual DOF charts rather
    than a single collapsed scalar.
    """
    info = copy.deepcopy(original_info)

    video_info = dict(DEFAULT_VIDEO_INFO)
    video_info['video.fps'] = fps

    total_videos = 0
    for col_name in image_columns:
        feat = info['features'].get(col_name)
        if feat is None:
            continue
        feat['dtype'] = 'video'
        feat['video_info'] = video_info
        total_videos += total_episodes

    info['total_videos'] = total_videos

    # --- Patch state / action motor names for the Panda robot ---
    # Maps feature key -> (expected shape, correct names dict)
    _motor_names_map: dict[str, tuple[list[int], dict]] = {
        'state': ([8], PANDA_EEF_STATE_NAMES),
        'actions': ([7], PANDA_EEF_ACTION_NAMES),
        # Also handle the "observation.state" / "action" naming variants
        'observation.state': ([8], PANDA_EEF_STATE_NAMES),
        'action': ([7], PANDA_EEF_ACTION_NAMES),
    }
    for feat_key, (expected_shape, names_dict) in _motor_names_map.items():
        feat = info['features'].get(feat_key)
        if feat is None:
            continue
        if feat.get('shape') == expected_shape:
            feat['names'] = names_dict

    return info


# ===================================================================
# Orchestrator
# ===================================================================
def prepare(
    dataset_dir: str,
    output_dir: str | None = None,
    *,
    force: bool = False,
    keep_images: bool = False,
    workers: int = 1,
) -> None:
    """Main entry: convert an image-mode LeRobot dataset to video mode."""
    _ensure_deps()

    # --- Locate dataset root & info.json ---
    meta_root, subset_prefix = find_dataset_root(dataset_dir)
    info_path = os.path.join(meta_root, 'meta', 'info.json')
    with open(info_path) as f:
        info = json.load(f)

    fps: int = info.get('fps', 10)
    total_episodes: int = info['total_episodes']
    chunks_size: int = info.get('chunks_size', 1000)
    data_path_template: str = info['data_path']
    video_path_template: str = info.get(
        'video_path',
        'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
    )

    # Identify image columns (dtype == "image")
    image_columns = [
        name
        for name, feat in info['features'].items()
        if feat.get('dtype') == 'image'
    ]

    if not image_columns:
        print(colored(
            'This dataset has no image features -- nothing to convert.',
            'yellow',
        ))
        return

    # --- Determine output directory ---
    if output_dir is None:
        output_dir = dataset_dir.rstrip('/') + '_viz'
    output_dir = os.path.normpath(output_dir)

    # If there is a subset prefix, output needs the same structure
    if subset_prefix:
        output_root = os.path.join(output_dir, subset_prefix)
    else:
        output_root = output_dir

    os.makedirs(output_root, exist_ok=True)

    print(
        colored(
            '\n  VLA-Arena LeRobot Visualizer Data Preprocessor\n',
            'cyan',
            attrs=['bold'],
        )
    )
    print(f'  Source dataset   : {dataset_dir}')
    if subset_prefix:
        print(f'  Subset prefix    : {subset_prefix}/')
    print(f'  Output directory : {output_dir}')
    print(f'  FPS              : {fps}')
    print(f'  Total episodes   : {total_episodes}')
    print(f'  Image columns    : {", ".join(image_columns)}')
    print(f'  Video codec      : H.264 ({H264_CODEC})')
    print(f'  Workers          : {workers}')
    print(f'  Strip images     : {not keep_images}')
    print(f'  Force reprocess  : {force}')
    print()

    # --- Phase 1: Convert episodes (parquet → MP4) ---
    print(colored(
        f'  [1/3] Encoding {total_episodes} episodes '
        f'× {len(image_columns)} cameras → MP4 ...',
        'cyan',
    ))

    t0 = time.time()

    # Build list of (episode_index, src_parquet_path) tuples
    episodes_to_process = []
    for ep_idx in range(total_episodes):
        ep_chunk = ep_idx // chunks_size
        rel_parquet = data_path_template.format(
            episode_chunk=ep_chunk,
            episode_index=ep_idx,
        )
        src_parquet = os.path.join(meta_root, rel_parquet)
        if os.path.isfile(src_parquet):
            episodes_to_process.append((ep_idx, src_parquet))

    total_videos_written = 0

    if workers > 1:
        # Parallel mode
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for ep_idx, src_pq in episodes_to_process:
                fut = executor.submit(
                    _process_one_episode,
                    src_pq,
                    image_columns,
                    video_path_template,
                    output_root,
                    ep_idx,
                    chunks_size,
                    fps,
                    force,
                )
                futures[fut] = ep_idx

            for i, fut in enumerate(as_completed(futures), 1):
                ep_idx = futures[fut]
                try:
                    result = fut.result()
                    total_videos_written += len(result)
                except Exception as exc:
                    print(
                        colored(
                            f'\n    Episode {ep_idx}: ERROR - {exc}',
                            'red',
                        )
                    )
                _print_progress(i, len(episodes_to_process))
    else:
        # Sequential mode
        for i, (ep_idx, src_pq) in enumerate(episodes_to_process, 1):
            try:
                result = _process_one_episode(
                    src_pq,
                    image_columns,
                    video_path_template,
                    output_root,
                    ep_idx,
                    chunks_size,
                    fps,
                    force,
                )
                total_videos_written += len(result)
            except Exception as exc:
                print(
                    colored(
                        f'\n    Episode {ep_idx}: ERROR - {exc}',
                        'red',
                    )
                )
            _print_progress(i, len(episodes_to_process))

    t1 = time.time()
    print(
        f'\n    Wrote {total_videos_written} video files '
        f'in {t1 - t0:.1f}s'
    )

    # --- Phase 2: Copy & optionally strip parquet data ---
    print(colored(
        '\n  [2/3] Copying data files (parquet) ...',
        'cyan',
    ))

    t2 = time.time()
    for i, (ep_idx, src_pq) in enumerate(episodes_to_process, 1):
        ep_chunk = ep_idx // chunks_size
        rel_parquet = data_path_template.format(
            episode_chunk=ep_chunk,
            episode_index=ep_idx,
        )
        dst_pq = os.path.join(output_root, rel_parquet)

        if os.path.isfile(dst_pq) and not force:
            _print_progress(i, len(episodes_to_process))
            continue

        if keep_images:
            # Just copy the file as-is
            import shutil
            os.makedirs(os.path.dirname(dst_pq), exist_ok=True)
            shutil.copy2(src_pq, dst_pq)
        else:
            _copy_parquet_without_images(src_pq, dst_pq, image_columns)
        _print_progress(i, len(episodes_to_process))

    t3 = time.time()
    print(f'\n    Copied {len(episodes_to_process)} parquet files in {t3 - t2:.1f}s')

    # --- Phase 3: Write updated metadata ---
    print(colored(
        '\n  [3/3] Writing updated metadata ...',
        'cyan',
    ))

    # Copy all metadata files first
    src_meta = os.path.join(meta_root, 'meta')
    dst_meta = os.path.join(output_root, 'meta')
    _copy_meta_dir(src_meta, dst_meta, exclude=['info.json'])

    # Write updated info.json
    viz_info = _build_viz_info(info, image_columns, fps, total_episodes)
    info_out = os.path.join(dst_meta, 'info.json')
    os.makedirs(dst_meta, exist_ok=True)
    with open(info_out, 'w') as f:
        json.dump(viz_info, f, indent=2)
        f.write('\n')
    print(f'    Wrote {info_out}')

    # --- Summary ---
    elapsed = time.time() - t0
    print(
        colored(
            f'\n  Done! Processed {len(episodes_to_process)} episodes '
            f'in {elapsed:.1f}s.\n',
            'green',
            attrs=['bold'],
        )
    )
    # Compute LOCAL_DATASET_DIR: the route expects LOCAL_DIR/{org}/{dataset}
    # so LOCAL_DIR is the parent-of-parent of output_dir.
    local_ds_dir = os.path.dirname(os.path.dirname(output_dir))
    dataset_name = os.path.basename(output_dir)
    org_name = os.path.basename(os.path.dirname(output_dir))

    print(
        colored(
            '  To use with the visualizer:\n',
            'green',
        )
    )
    print(
        f'    cd <visualizer-dir>\n'
        f'    LOCAL_DATASET_DIR={local_ds_dir} bun run dev\n'
    )
    print(
        f'  Then open: http://localhost:3000/{org_name}/{dataset_name}/0\n'
    )


# ===================================================================
# Helpers
# ===================================================================
def _copy_meta_dir(src: str, dst: str, exclude: list[str] | None = None):
    """Recursively copy metadata directory, skipping files in *exclude*."""
    import shutil

    exclude = exclude or []
    if not os.path.isdir(src):
        return

    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        dst_dir = os.path.join(dst, rel) if rel != '.' else dst
        os.makedirs(dst_dir, exist_ok=True)

        for fn in filenames:
            if fn in exclude and rel == '.':
                continue
            src_file = os.path.join(dirpath, fn)
            dst_file = os.path.join(dst_dir, fn)
            shutil.copy2(src_file, dst_file)


def _print_progress(current: int, total: int) -> None:
    """Print an inline progress indicator."""
    pct = current / total * 100 if total else 100
    bar_len = 30
    filled = int(bar_len * current // total) if total else bar_len
    bar = '█' * filled + '░' * (bar_len - filled)
    print(
        f'\r    [{bar}] {current}/{total} ({pct:.0f}%)',
        end='',
        flush=True,
    )


# ===================================================================
# CLI entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            'Convert VLA-Arena LeRobot datasets from image mode '
            '(PNG frames in Parquet) to video mode (MP4 files) for '
            'smooth playback in the lerobot-dataset-visualizer.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process the minimum-viable L0_S dataset
  %(prog)s ~/.cache/huggingface/lerobot/VLA-Arena/VLA_Arena_L0_S_lerobot_openpi

  # Process with a custom output directory
  %(prog)s /data/dataset --output-dir /data/dataset_viz

  # Re-process everything, keeping original image columns
  %(prog)s /data/dataset --force --keep-images

  # Parallel processing (4 workers)
  %(prog)s /data/dataset --workers 4
        """,
    )

    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Path to the downloaded LeRobot dataset directory',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=(
            'Output directory for the visualizer-ready dataset '
            '(default: <dataset_dir>_viz)'
        ),
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-process episodes even if output already exists',
    )
    parser.add_argument(
        '--keep-images',
        action='store_true',
        help='Keep image columns in the output parquet files',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)',
    )

    args = parser.parse_args()

    prepare(
        args.dataset_dir,
        args.output_dir,
        force=args.force,
        keep_images=args.keep_images,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
