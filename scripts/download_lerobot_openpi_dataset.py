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
VLA-Arena LeRobot OpenPi Dataset Downloader

Discover, inspect, and download LeRobot-format datasets published by the
VLA-Arena organisation on HuggingFace Hub for the OpenPi (Pi-Zero) model.

Datasets are discovered dynamically via the HuggingFace API -- no hardcoded
list required.  

The download uses `huggingface_hub.snapshot_download()`, the
same mechanism that LeRobotDataset.pull_from_repo() calls internally in the
OpenPi training pipeline.

Usage:
    # List all available openpi LeRobot datasets on HuggingFace
    vla-arena.download-lerobot-openpi list

    # Show detailed metadata for a specific dataset
    vla-arena.download-lerobot-openpi info VLA-Arena/VLA_Arena_L0_M_lerobot_openpi

    # Download a dataset to the default cache location
    vla-arena.download-lerobot-openpi download VLA-Arena/VLA_Arena_L0_M_lerobot_openpi

    # Download to a custom directory
    vla-arena.download-lerobot-openpi download VLA-Arena/VLA_Arena_L0_S_lerobot_openpi --local-dir /data/vla
"""

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure huggingface_hub is importable (may live in openpi's .venv)
# ---------------------------------------------------------------------------
_HF_HUB_READY = False


def _ensure_hf_hub():
    """Make huggingface_hub importable.

    Checks the standard Python path first, then falls back to the
    site-packages inside openpi's uv-managed .venv.
    """
    global _HF_HUB_READY
    if _HF_HUB_READY:
        return

    try:
        import huggingface_hub  # noqa: F401

        _HF_HUB_READY = True
        return
    except ImportError:
        pass

    import glob

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
    for site_dir in glob.glob(venv_pattern):
        site_dir = os.path.normpath(site_dir)
        if site_dir not in sys.path:
            sys.path.insert(0, site_dir)

    try:
        import huggingface_hub  # noqa: F401

        _HF_HUB_READY = True
    except ImportError:
        print(
            'Error: Could not import huggingface_hub.\n'
            'Install it with one of:\n'
            '  pip install huggingface_hub\n'
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
HF_ORG = 'VLA-Arena'
DATASET_SUFFIX = 'lerobot_openpi'

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser('~'), '.cache', 'huggingface', 'lerobot'
)


# ===================================================================
# Core utility: discover openpi LeRobot datasets from HuggingFace
# ===================================================================
def fetch_openpi_datasets():
    """Fetch all HuggingFace datasets under *VLA-Arena* whose name ends
    with ``lerobot_openpi``.

    Returns a list of ``huggingface_hub.DatasetInfo`` objects sorted by
    dataset id, so the caller can iterate / filter further.
    """
    _ensure_hf_hub()
    from huggingface_hub import HfApi

    api = HfApi()
    all_datasets = list(api.list_datasets(author=HF_ORG))
    matched = [
        ds for ds in all_datasets if ds.id.endswith(DATASET_SUFFIX)
    ]
    matched.sort(key=lambda d: d.id)
    return matched


# ===================================================================
# Helpers
# ===================================================================
def _resolve_local_dir(repo_id, local_dir=None):
    """Return the local cache directory for *repo_id*."""
    if local_dir:
        return local_dir
    return os.path.join(DEFAULT_CACHE_DIR, repo_id)


def _parse_dataset_name(repo_id):
    """Extract human-readable components from a repo_id.

    E.g. ``VLA-Arena/VLA_Arena_L0_M_lerobot_openpi``
    -> level=L0, size=M
    """
    name = repo_id.split('/')[-1]  # VLA_Arena_L0_M_lerobot_openpi
    parts = name.split('_')
    # Expected: ['VLA', 'Arena', 'L0', 'M', 'lerobot', 'openpi']
    level = parts[2] if len(parts) > 2 else '?'
    size_code = parts[3] if len(parts) > 3 else '?'
    size_map = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    size_label = size_map.get(size_code, size_code)
    return level, size_code, size_label


def _fetch_info_json(repo_id):
    """Download and parse *info.json* (and optionally *tasks.jsonl*) from
    a LeRobot dataset repository on HuggingFace.

    Handles v2 format where metadata lives under a subset prefix, e.g.
    ``VLA_Arena/meta/info.json``.
    """
    _ensure_hf_hub()
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type='dataset')

    info_paths = [f for f in files if f.endswith('meta/info.json')]
    if not info_paths:
        return None, None

    info_path = info_paths[0]
    local_path = hf_hub_download(repo_id, info_path, repo_type='dataset')
    with open(local_path) as fh:
        info = json.load(fh)

    # Optionally load tasks.jsonl living next to info.json
    tasks_path = info_path.replace('info.json', 'tasks.jsonl')
    tasks = {}
    try:
        local_tasks = hf_hub_download(
            repo_id, tasks_path, repo_type='dataset'
        )
        with open(local_tasks) as fh:
            for line in fh:
                entry = json.loads(line)
                tasks[entry['task_index']] = entry['task']
    except Exception:
        pass

    info['_tasks'] = tasks
    return info, info_path


# ===================================================================
# Sub-command: list
# ===================================================================
def cmd_list(args):
    """List all openpi LeRobot datasets published by VLA-Arena."""
    datasets = fetch_openpi_datasets()

    if not datasets:
        print(colored('\nNo datasets found.\n', 'yellow'))
        return

    print(
        colored(
            f'\nFound {len(datasets)} openpi LeRobot dataset(s) '
            f'under "{HF_ORG}":\n',
            'green',
        )
    )

    # Table header
    hdr_fmt = '  {:<55s} {:>5s} {:>5s} {:>10s}  {}'
    row_fmt = '  {:<55s} {:>5s} {:>5s} {:>10s}  {}'
    print(
        colored(
            hdr_fmt.format('REPO ID', 'LEVEL', 'SIZE', 'DOWNLOADS', 'LAST MODIFIED'),
            attrs=['bold'],
        )
    )
    print('  ' + '-' * 100)

    for ds in datasets:
        level, size_code, _ = _parse_dataset_name(ds.id)
        downloads = str(ds.downloads) if ds.downloads is not None else '?'
        modified = (
            ds.last_modified.strftime('%Y-%m-%d')
            if ds.last_modified
            else '?'
        )
        print(row_fmt.format(ds.id, level, size_code, downloads, modified))

    print(
        colored(
            f'\nTip: Use "info <REPO_ID>" to see detailed metadata, '
            f'or "download <REPO_ID>" to fetch data.\n',
            'cyan',
        )
    )


# ===================================================================
# Sub-command: info
# ===================================================================
def cmd_info(args):
    """Show detailed metadata for a specific dataset."""
    repo_id = args.repo_id

    print(f'\nDataset info: {repo_id}')
    print('=' * 72)

    try:
        info, info_path = _fetch_info_json(repo_id)
    except Exception as e:
        print(colored(f'  Error: {e}', 'red'))
        return

    if info is None:
        print(colored('  No info.json found in this repository.', 'yellow'))
        return

    local_dir = _resolve_local_dir(repo_id)
    local_exists = os.path.isdir(local_dir)

    print(f'  Repo ID          : {repo_id}')
    print(
        f'  HuggingFace URL  : https://huggingface.co/datasets/{repo_id}'
    )
    print(
        f'  Local cache      : {local_dir} '
        f'({"exists" if local_exists else "not downloaded"})'
    )
    print(f'  Metadata path    : {info_path}')
    print(f'  Codebase version : {info.get("codebase_version", "N/A")}')
    print(f'  Data path        : {info.get("data_path", "N/A")}')
    print(f'  FPS              : {info.get("fps", "N/A")}')
    print(f'  Total episodes   : {info.get("total_episodes", "N/A")}')
    print(f'  Total frames     : {info.get("total_frames", "N/A")}')
    print(f'  Total tasks      : {info.get("total_tasks", "N/A")}')
    print(f'  Chunks size      : {info.get("chunks_size", "N/A")}')
    print(f'  Total chunks     : {info.get("total_chunks", "N/A")}')

    # Features
    features = info.get('features', {})
    if features:
        print('\n  Features:')
        for name, feat in features.items():
            dtype = feat.get('dtype', feat.get('_type', '?'))
            shape = feat.get('shape', '')
            names = feat.get('names', '')
            if shape:
                print(
                    f'    {name:24s}  {dtype}  '
                    f'shape={shape}  names={names}'
                )
            else:
                print(f'    {name:24s}  {dtype}')

    # Tasks
    tasks = info.get('_tasks', {})
    if tasks:
        print(f'\n  Tasks ({len(tasks)}):')
        for idx, task_desc in list(tasks.items())[:10]:
            print(f'    [{idx:>3}] {task_desc}')
        if len(tasks) > 10:
            print(f'    ... and {len(tasks) - 10} more')

    print()


# ===================================================================
# Sub-command: download
# ===================================================================
def cmd_download(args):
    """Download a dataset to the local cache."""
    _ensure_hf_hub()
    from huggingface_hub import snapshot_download

    repo_id = args.repo_id
    local_dir = _resolve_local_dir(repo_id, args.local_dir)

    print(f'\nDownloading LeRobot dataset: {repo_id}')
    print('=' * 72)

    # ---- Step 1: Fetch & display metadata ----
    print(colored('\n[1/2] Fetching dataset metadata...', 'cyan'))
    try:
        info, _ = _fetch_info_json(repo_id)
    except Exception as e:
        print(colored(f'\n  Error fetching metadata: {e}', 'red'))
        print(
            f'\n  Verify the repo: '
            f'https://huggingface.co/datasets/{repo_id}'
        )
        sys.exit(1)

    if info:
        print(f'  Codebase version : {info.get("codebase_version", "N/A")}')
        print(f'  FPS              : {info.get("fps", "N/A")}')
        print(f'  Total episodes   : {info.get("total_episodes", "N/A")}')
        print(f'  Total frames     : {info.get("total_frames", "N/A")}')
        print(f'  Total tasks      : {info.get("total_tasks", "N/A")}')
        print(f'  Chunks size      : {info.get("chunks_size", "N/A")}')
    else:
        print('  (No info.json found -- will download all files)')

    # ---- Step 2: Snapshot download ----
    print(
        colored(
            '\n[2/2] Downloading dataset files '
            '(this may take a while)...',
            'cyan',
        )
    )
    print(f'  Destination: {local_dir}')
    start = time.time()
    try:
        snapshot_download(
            repo_id,
            repo_type='dataset',
            local_dir=local_dir,
            token=args.token,
        )
    except Exception as e:
        print(colored(f'\n  Error downloading data: {e}', 'red'))
        sys.exit(1)

    elapsed = time.time() - start
    print(f'  Elapsed: {elapsed:.1f}s')

    print(colored(f'\n  Dataset cached at: {local_dir}', 'green'))
    print(colored('  Download complete!\n', 'green'))


# ===================================================================
# CLI entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            'Discover, inspect, and download VLA-Arena LeRobot datasets '
            'for the OpenPi (Pi-Zero) model.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  %(prog)s list

  # Show metadata for a specific dataset
  %(prog)s info VLA-Arena/VLA_Arena_L0_M_lerobot_openpi

  # Download a dataset
  %(prog)s download VLA-Arena/VLA_Arena_L0_S_lerobot_openpi

  # Download to a custom location
  %(prog)s download VLA-Arena/VLA_Arena_L0_S_lerobot_openpi --local-dir /data/vla
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # ---- list ----
    subparsers.add_parser(
        'list',
        help=(
            f'List all "{DATASET_SUFFIX}" datasets '
            f'published by {HF_ORG}'
        ),
    )

    # ---- info ----
    info_parser = subparsers.add_parser(
        'info',
        help='Show detailed metadata for a dataset',
    )
    info_parser.add_argument(
        'repo_id',
        help='HuggingFace dataset repo_id '
        '(e.g. VLA-Arena/VLA_Arena_L0_M_lerobot_openpi)',
    )

    # ---- download ----
    dl_parser = subparsers.add_parser(
        'download',
        help='Download a dataset to the local cache',
    )
    dl_parser.add_argument(
        'repo_id',
        help='HuggingFace dataset repo_id '
        '(e.g. VLA-Arena/VLA_Arena_L0_M_lerobot_openpi)',
    )
    dl_parser.add_argument(
        '--local-dir',
        type=str,
        default=None,
        help=(
            'Local directory to save dataset '
            f'(default: {DEFAULT_CACHE_DIR}/<repo_id>)'
        ),
    )
    dl_parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace API token (for private repos)',
    )

    args = parser.parse_args()

    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'download':
        cmd_download(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
