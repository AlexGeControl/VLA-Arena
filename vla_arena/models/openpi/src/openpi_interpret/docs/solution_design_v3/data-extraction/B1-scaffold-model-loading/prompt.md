# Task B1: Scaffold — Pi-Zero load + one inference

> Part of the [Data Extraction epic](../README.md). No dependency on B2–B5; this task establishes the runnable harness.

## Goal

Deliver a **CLI script** that:

1. Loads the fine-tuned **Pi-Zero** policy from a **local** checkpoint directory.
2. Loads **one episode’s** frames / metadata from the LeRobot-style HF dataset (metadata + parquet, **local only**).
3. Runs **at least one** forward / inference step successfully (policy output actions or internal tensors as needed for the next tasks).

## Critical pitfalls

### PITFALL #1 — HF checkpoint corruption

- **Do not** rely on `huggingface_hub.snapshot_download` for the Pi-Zero checkpoint if it has produced corrupt trees in your environment.
- Use the **direct snapshot directory** under the HF cache, for example:

  `~/.cache/huggingface/hub/models--VLA-Arena--pi0-vla-arena-fintuned/snapshots/acdc8e7eaa6dfccedef6db26626ec828bfa21b1e`

- Pass that path as the checkpoint root and use Hugging Face APIs with **`local_files_only=True`** where applicable.

### PITFALL #12 — Policy object attribute

- The trainable JAX module is exposed as **`policy._model`** (private attribute), **not** `policy.model`.
- Any code that needs Gemma / Pi-Zero parameters or submodules must go through **`policy._model`** (or the project’s documented accessor if one exists).

## Model loading

Use OpenPI’s policy factory:

```python
from openpi.policies.policy_config import create_trained_policy

policy = create_trained_policy(config, checkpoint_dir)
```

- **`config`**: `pi0_vla_arena_low_mem_finetune` (or the exact config name string / object used in this repo for L0 fine-tuned Pi-Zero).
- **`checkpoint_dir`**: filesystem path to the snapshot (see PITFALL #1).

## Dataset access

- **`dataset-repo-id`**: default e.g. `VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla` (or team default).
- Use `huggingface_hub.hf_hub_download` (or equivalent) with **`local_files_only=True`** for:
  - Dataset **metadata** / info files
  - **Parquet** shards needed to iterate frames

Pre-download once with the CLI if needed; B1 should fail fast with a clear message if files are missing locally.

## CLI arguments

| Arg | Purpose | Default suggestion |
|-----|---------|-------------------|
| `--checkpoint` | Directory passed to `create_trained_policy` | Known-good cache snapshot path (PITFALL #1) |
| `--dataset-repo-id` | HF dataset id | `VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla` |
| `--output-dir` | Where to write logs / future HDF5 | Required or `./outputs/extraction` |
| `--max-episodes` | Cap episodes processed | `3` |
| `--timestep-stride` | Frame stride along episode | `10` |

Implement in **`extract_interpret_data.py`** (skeleton) or a thin **`b1_smoke.py`** that you merge later—prefer one entrypoint documented in the epic README.

## Minimal inference loop (sketch)

```python
# Pseudocode — align with openpi Policy API in this repo
obs = build_obs_from_lerobot_row(row, images)
action_chunk = policy.infer(obs)  # or sample_actions / infer API as defined locally
```

Use the **actual** observation dict keys and image layout expected by `policy.infer` in OpenPI (match training / eval).

## Acceptance criteria

- [ ] CLI parses args; defaults match the table (or documented team overrides).
- [ ] `create_trained_policy` succeeds with **`local_files_only`** checkpoint path; no `snapshot_download` in the critical path for that checkpoint.
- [ ] Policy internals accessed via **`policy._model`** only (not `policy.model`).
- [ ] At least **one** dataset row / frame loads with **`hf_hub_download(..., local_files_only=True)`**.
- [ ] **One** inference call completes without exception on GPU (or documented CPU fallback for CI-only, if the team allows it).
- [ ] Logging prints episode id, frame index, and checkpoint path for reproducibility.
