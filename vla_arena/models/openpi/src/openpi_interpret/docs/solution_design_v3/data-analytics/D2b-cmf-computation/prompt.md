# Task D2b: CMF score computation

> Part of the [Data Analytics epic](../README.md). Depends on [D1](../D1-analytics-scaffold/prompt.md) and [D2a](../D2a-v-capture-cmf-attended/prompt.md).

## Goal

Implement **`analytics/cmf.py`** that reads pre-computed CMF attended representations and Q-projections from HDF5, computes per-head cosine similarity in 256-d, and aggregates across heads and queries.

## The 5 CMF pairs

| Pair | Query embedding | Attended target | Notes |
|------|----------------|-----------------|-------|
| S → L | `q_suffix[0, h, :]` | `attended_language[0, h, :]` | Single query (state) |
| S → V | `q_suffix[0, h, :]` | `attended_visual[0, h, :]` | Single query (state) |
| A → L | `q_suffix[1:51, h, :]` | `attended_language[1:51, h, :]` | 50 action queries |
| A → V | `q_suffix[1:51, h, :]` | `attended_visual[1:51, h, :]` | 50 action queries |
| A → S | `q_suffix[1:51, h, :]` | `q_suffix[0, h, :]` | Direct cosine, degenerate case |

**A → S note**: With 1 target token, normalized attention = 1, so the attended representation equals the target embedding. CMF reduces to direct cosine similarity between action and state Q-projections. No pre-computed attended representation needed.

## Core function

```python
def compute_all_cmf(
    q_suffix: np.ndarray,            # [51, 8, 256]
    attended_language: np.ndarray,    # [51, 8, 256]
    attended_visual: np.ndarray,      # [51, 8, 256]
) -> TimestepCmf:
```

## Per-head cosine similarity

For each head h, for each query index k:

$$\text{CMF}^{(h)}(k) = \frac{q_k^{(h)} \cdot v_{\text{attended},k}^{(h)}}{\|q_k^{(h)}\| \, \|v_{\text{attended},k}^{(h)}\|}$$

Guard zero-norm vectors with `clip(min=epsilon)`.

## Aggregation chain

```
Per head (cosine in R^256)
    → mean over 8 heads
        → mean over query tokens (50 for A→X, 1 for S→X)
            = single scalar per CMF pair per timestep
```

Episode-level and global aggregation happen in the CLI orchestrator (D4).

## File placement

**`openpi_interpret/analytics/cmf.py`**

Key functions:
- `compute_all_cmf(q_suffix, attended_language, attended_visual) -> TimestepCmf`
- `_cmf_from_attended(q_suffix, attended, query_indices) -> float`
- `_cmf_direct(q_suffix, query_indices, target_index) -> float` (A → S)
- `_batched_cosine_mean(a, b) -> float`

## Test cases (`tests/test_cmf.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_identical_vectors_give_one` | `a == b` | CMF ≈ 1.0 |
| `test_orthogonal_vectors_give_zero` | `a ⊥ b` | CMF ≈ 0.0 |
| `test_opposite_vectors_give_negative_one` | `a == -b` | CMF ≈ −1.0 |
| `test_aligned_attended_gives_high_cmf` | attended = q_suffix copy | CMF > 0.99 |
| `test_random_gives_near_zero` | random a, b | −0.3 < CMF < 0.3 |
| `test_returns_all_five_pairs` | random inputs | all 5 fields present, in [−1, 1] |
| `test_v_shared_across_heads` | identical attention across heads | attended representations identical across heads |

## Acceptance criteria

- [ ] `compute_all_cmf` returns `TimestepCmf` with all 5 float fields.
- [ ] All values in `[-1.0, 1.0]`.
- [ ] A → S computed via direct cosine (no cmf_attended dependency).
- [ ] Zero-norm vectors handled gracefully (no NaN, no crash).
- [ ] All tests pass.
