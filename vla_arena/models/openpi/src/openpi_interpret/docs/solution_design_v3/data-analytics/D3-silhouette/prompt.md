# Task D3: Silhouette Coefficient

> Part of the [Data Analytics epic](../README.md). Depends on [D1](../D1-analytics-scaffold/prompt.md).

## Goal

Implement **`analytics/silhouette.py`** that computes the Silhouette Coefficient on t-SNE projections using a 3-group modality clustering that reflects the Pi-Zero dual-expert architecture.

## Cluster definition

| Cluster | Label | Included modalities | Token range | Count |
|---------|-------|--------------------| ------------|-------|
| 0 (Visual) | `visual` | `base_0_rgb`, `left_wrist_0_rgb` | 0–511 | 512 |
| 1 (Language) | `language` | `language` | 768–815 | 48 |
| 2 (Action) | `action` | `state`, `action` | 816–866 | 51 |

**Excluded**: `right_wrist_0_rgb` (tokens 512–767) — zero placeholder for Pi-Zero on VLA-Arena. Including 256 noise tokens from a dummy camera degrades cluster quality.

**Rationale for 3 groups**:
- Right wrist excluded because it carries no visual information.
- State merged with action because both are Expert 1 tokens projected through the same Q/K/V weights from the same 1024-d hidden space.

## Implementation

```python
from sklearn.metrics import silhouette_score

def build_modality_labels() -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels [867], mask [867] bool)."""
    # labels: integer 0/1/2 for each of 867 tokens
    # mask: True for included tokens, False for right wrist (512-767)

def compute_silhouette(tsne_coords: np.ndarray) -> TimestepSilhouette:
    """Compute silhouette on masked t-SNE coords."""
    labels, mask = build_modality_labels()
    score = silhouette_score(tsne_coords[mask], labels[mask], metric="euclidean")
    return TimestepSilhouette(score=float(score))
```

Cache the labels/mask arrays (they're deterministic).

## Silhouette score interpretation

| Range | Interpretation |
|-------|----------------|
| 0.71 – 1.00 | Strong structure — modalities in distinct regions |
| 0.51 – 0.70 | Reasonable structure — distinguishable with overlap |
| 0.26 – 0.50 | Weak structure — clusters exist but significantly overlap |
| ≤ 0.25 | No meaningful structure — modalities interleaved |

Expected range for Pi-Zero: **0.30–0.40** (weak structure; expected since cross-expert attention requires modality overlap in the shared head space).

## File placement

**`openpi_interpret/analytics/silhouette.py`**

## Test cases (`tests/test_silhouette.py`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_length` | labels/mask shapes | `(867,)` each |
| `test_three_unique_labels` | unique labels in masked region | 3 |
| `test_right_wrist_excluded` | mask[512:768] | all False |
| `test_included_count` | mask.sum() | 512 + 48 + 51 = 611 |
| `test_well_separated_clusters` | synthetic separated coords | score > 0.5 |
| `test_random_coords` | random coords | score in [−1, 1] |

## Acceptance criteria

- [ ] Mask excludes exactly tokens 512–767 (right wrist).
- [ ] 611 tokens included (512 visual + 48 language + 51 action+state).
- [ ] 3 unique integer labels in the masked region.
- [ ] `sklearn.metrics.silhouette_score` called with `metric="euclidean"`.
- [ ] Output is `TimestepSilhouette` with score in `[-1, 1]`.
- [ ] Labels/mask cached after first call (no recomputation).
- [ ] All tests pass.
