# Task B1: Script Scaffold & Model Loading

> Part of the [Data Extraction Pipeline](../README.md) epic.

## Goal

A runnable script that loads the Pi-Zero checkpoint and can perform a single inference on one observation from the L0 S dataset.

## Task

Create the extraction script at:

```
openpi_interpret/extraction/extract_interpret_data.py
```

### CLI Interface

Using `argparse` or `tyro`:

```bash
python extract_interpret_data.py \
  --checkpoint "$CKPT_DIR" \
  --dataset-repo-id "VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla" \
  --output-dir "../data" \
  --max-episodes 10 \
  --timestep-stride 30 \
  --denoising-capture-step -1
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to fine-tuned checkpoint directory |
| `--dataset-repo-id` | `VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla` | HuggingFace dataset repo |
| `--output-dir` | `../data` | Output directory for binary files |
| `--max-episodes` | `10` | Number of episodes to process |
| `--episode-ids` | `None` | Specific episode indices (comma-separated) |
| `--timestep-stride` | `30` | Sample every Nth frame per episode |
| `--denoising-capture-step` | `-1` | Which denoising step to capture (-1 = last) |
| `--split` | `train` | HF dataset split |

### Model Loading

Mirrors `serve_policy.py` and training pipeline:

```python
from openpi.training.config import get_config
from openpi.models.pi0 import Pi0

config = get_config("pi0_vla_arena_low_mem_finetune")

# Initialize model with a dummy batch for shape inference
model = config.model.create(init_rng, sample_batch)

# Load fine-tuned weights
train_state = TrainState(model=model, ...)
config.weight_loader.load(train_state)
```

### Dataset Loading

Mirrors `qual_analysis/runner.py`:

```python
from datasets import load_dataset

hf_ds = load_dataset(dataset_repo_id, split=split)

# Group frames by episode (same as runner.py)
episode_groups = _group_by_episode_fast(hf_ds)

# Select episodes
selected = sorted(episode_groups.keys())[:max_episodes]
```

### Observation Construction

Mirrors `OpenPiBaseAdapter` + `LiberoInputs`:

```python
for ep_id in selected:
    row_indices = episode_groups[ep_id]
    sampled_indices = row_indices[::timestep_stride]

    for row_idx in sampled_indices:
        sample = hf_ds[row_idx]
        observation = build_observation(sample)  # agent_image, wrist_image, state
        model_inputs = apply_transforms(observation, task_description)
        # ... run inference with capture hooks (Task B2) ...
```

### Output File

```
openpi_interpret/extraction/extract_interpret_data.py
```

## Acceptance Criteria

- [ ] Script loads the checkpoint without errors
- [ ] Can run `sample_actions` on one observation and get a valid action trajectory
- [ ] CLI arguments are parsed and validated
- [ ] Dataset loads and episode grouping works
