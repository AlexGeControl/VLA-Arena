# Cross-Modal Fusion Analysis — Pi-Zero on VLA-Arena

Analysis of cross-modal integration in the Pi-Zero dual-expert Transformer,
measured via **Cross-Modal Fusion (CMF)** scores and **Silhouette Coefficients**
on the VLA-Arena L0 S dataset.

> **Model**: Pi-Zero (Pi0), fine-tuned on VLA-Arena L0
> **Dataset**: `VLA-Arena/VLA_Arena_L0_S_lerobot_smolvla` (5 episodes, 64 timesteps)
> **Layer**: 17 (final Gemma backbone layer)
> **CMF embedding**: V-projections (exact, single KV head, 256-d)
> **Query embedding**: Q-projections (per-head, 8 × 256-d)

## Metric Definitions

### Cross-Modal Fusion (CMF)

Attention-weighted cross-modal cosine similarity ([cmf.tex](cmf.tex)).
For each query token $q_k$ attending to a set of target tokens $\{v_j\}$
in modality $M$:

1. Normalize attention within modality: $\hat{\alpha}_{k,j} = \alpha_{k,j} / \sum_{j' \in M} \alpha_{k,j'}$
2. Compute attended representation: $v_{\text{attended},k} = \sum_{j \in M} \hat{\alpha}_{k,j} \cdot v_j$
3. Cosine similarity: $\text{CMF}(k) = \cos(q_k, v_{\text{attended},k})$

Aggregation: per head (256-d cosine) → mean over 8 heads → mean over query tokens → mean over timesteps → mean over episodes.

### Silhouette Coefficient

Mean [Silhouette Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
on the 2-D t-SNE projection of the last layer's Q-projections.

Cluster definition (3 groups):
- **Visual**: base camera + left wrist camera (512 tokens; right wrist excluded — zero placeholder for Pi-Zero on VLA-Arena)
- **Language**: instruction tokens (48 tokens)
- **Action**: proprioceptive state + action tokens (51 tokens; both Expert 1)

## Results

### Global Scores

| CMF Pair | Query | Target | Score |
|----------|-------|--------|-------|
| A → S | Action tokens (Expert 1) | State token (Expert 1) | **0.2487** |
| A → L | Action tokens (Expert 1) | Language tokens (Expert 0) | 0.0024 |
| A → V | Action tokens (Expert 1) | Visual tokens (Expert 0) | −0.0181 |
| S → L | State token (Expert 1) | Language tokens (Expert 0) | −0.0221 |
| S → V | State token (Expert 1) | Visual tokens (Expert 0) | −0.0329 |

| Metric | Score |
|--------|-------|
| Silhouette Coefficient (3-group) | **0.3700** |

### Per-Episode Breakdown

| Episode | Timesteps | A→S | A→V | A→L | S→V | S→L | Silhouette |
|---------|-----------|------|------|------|------|------|------------|
| ep_000000 | 13 | 0.244 | −0.016 | 0.013 | −0.032 | −0.018 | 0.357 |
| ep_000001 | 11 | 0.253 | −0.008 | 0.003 | −0.027 | −0.013 | 0.383 |
| ep_000002 | 13 | 0.244 | −0.027 | 0.004 | −0.038 | −0.029 | 0.358 |
| ep_000003 | 15 | 0.258 | −0.019 | −0.008 | −0.036 | −0.027 | 0.371 |
| ep_000004 | 12 | 0.244 | −0.021 | 0.001 | −0.032 | −0.024 | 0.381 |

## Interpretation

### A → S is the dominant fusion pathway (0.25)

Action query tokens strongly align with the proprioceptive state token in the
shared 256-d head space. This is the only CMF pair with a meaningfully positive
score. It reflects the architectural reality that both state and action tokens
are processed by **Expert 1** (1024-d width) through the **same** Q/K/V
projection weights, naturally sharing representational structure.

### Cross-expert CMF is near-zero

All four cross-expert pairs (A→V, A→L, S→V, S→L) produce CMF scores
indistinguishable from zero (range: −0.033 to +0.002). This means the
Q-projections of Expert 1 tokens (action/state) and the V-projections of
Expert 0 tokens (visual/language) occupy **orthogonal subspaces** of the
shared 256-d head space.

### Near-zero CMF does not imply poor performance

Pi-Zero is one of the strongest VLA models on VLA-Arena despite near-zero
cross-expert CMF. This is not a contradiction — the attention mechanism
does not require Q–V cosine alignment to transfer information effectively:

1. **Attention routes, not aligns.** The Q·K dot product determines *which*
   tokens to attend to. The V projections carry *what* information to read.
   These are different learned linear maps optimized for different roles.
   Effective routing (high attention on relevant tokens) does not require
   the query to be directionally aligned with the value.

2. **The output projection compensates.** After attention computes
   `α @ V → [8, 256]`, a learned per-expert output projection
   (`attn_vec_einsum_1: [8, 256] → [1024]`) rotates and combines the
   attended representation into Expert 1's hidden space. This projection
   is the "decoder ring" trained end-to-end to extract useful information
   from V-space representations that may be orthogonal to Q-space.

3. **Dual-expert separation is by design.** Expert 0 (PaliGemma, 2048-d)
   and Expert 1 (action expert, 1024-d) have **separate** Q/K/V
   projection weights initialized from different sources. There is no
   training signal encouraging Q–V cosine alignment across experts —
   only the Q·K product needs to be discriminative for correct routing.

4. **The modality gap is a known phenomenon.** Near-zero cross-modal
   cosine similarity is consistent with the well-documented modality gap
   in multimodal representations (Liang et al., 2022; cited in
   [cmf.tex](cmf.tex)). The gap does not prevent effective information
   transfer through learned projection heads.

### Silhouette indicates partial cluster structure (0.37)

The t-SNE projection shows moderate separation between visual, language,
and action/state token groups. This falls in the "weak structure" band
(0.26–0.50), indicating that modality clusters are distinguishable but
significantly overlap — consistent with a shared attention space where
different modalities must interact.

## Methodology Notes

### Q-approximation vs V-projection

Initial CMF scores were computed using Q-projections for both query and
target embeddings ("Q-approximation"). Switching to exact V-projections
for the target side produced a significant shift:

| CMF Pair | Q-approx (3 ep) | V-exact (5 ep) |
|----------|-----------------|----------------|
| A → S | 0.250 | 0.249 |
| S → V | 0.069 | −0.033 |
| A → V | 0.048 | −0.018 |
| A → L | 0.018 | 0.002 |
| S → L | −0.005 | −0.022 |

The Q-approximation inflated cross-expert CMF because Q-projections have
8 heads per token (shared dimensionality across experts), creating
apparent overlap. V-projections have a single KV head and use different
learned weights, revealing the true cross-expert alignment gap.

### Silhouette cluster refinement

The silhouette score improved from 0.330 (6-group) to 0.370 (3-group)
after:
- Excluding right wrist camera tokens (zero placeholder for Pi-Zero)
- Merging state + action tokens (both Expert 1, same projection weights)

## Reproducibility

```bash
# Extract model states (GPU required, ~17 min for 5 episodes)
cd <openpi_root>
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  .venv/bin/python src/openpi_interpret/extraction/extract_interpret_data.py \
    --max-episodes 5 --timestep-stride 10

# Run analytics (CPU only, ~3 seconds)
conda run -n openpi-vla-arena python \
  src/openpi_interpret/analytics/run_analytics.py \
    --data-dir src/openpi_interpret/data \
    --output analytics_report.yaml --layer 17
```

Full per-timestep data: [`analytics_report.yaml`](../../../data/analytics_report.yaml)
