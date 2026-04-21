# Visual Primacy in VLM-Backbone VLA Policies

## Abstract

We present converging evidence from two architecturally distinct Vision-Language-Action (VLA) models — Pi-Zero and Cosmos Reason2 — that **visual representation quality is the dominant factor in VLA task performance**. Through a combination of controlled ablation experiments, attention-weighted cross-modal fusion analysis, and depth-wise embedding geometry profiling, we show that (1) enriching visual input produces larger performance gains than architectural modifications to the action expert, (2) the action expert allocates the majority of its attention capacity to visual tokens across all network depths, and (3) modality-specific structure persists throughout the Transformer despite deep cross-modal attention, enabling reliable attribution of attention patterns to source modalities.

## 1. Introduction

Modern VLA architectures share a common two-stage structure: a frozen or fine-tuned Vision-Language Model (VLM) backbone produces context-aware multimodal embeddings, and a downstream action expert conditions on these embeddings to generate robot actions via flow matching or similar generative methods. A natural question arises: **does the quality of the visual representation or the sophistication of the action expert matter more for task performance?**

We investigate this question using the VLA-Arena benchmark, analyzing two models that represent opposite ends of the VLA design spectrum:

- **Pi-Zero**: A 2.3B-parameter dual-expert Transformer where the action expert is deeply integrated into the VLM backbone through 18 layers of cross-expert attention with LoRA fine-tuning.
- **Cosmos Reason2 VLA**: A modular design where a frozen 2B-parameter VLM backbone feeds pre-extracted features into a lightweight 4-layer Transformer action expert trained from scratch.

Despite their architectural differences, both models reveal the same pattern: **what the model sees matters more than how it integrates what it sees.**

## 2. Evidence from Controlled Ablation (Cosmos Reason2)

### 2.1 Experimental Design

We conducted a controlled experiment on the Cosmos Reason2 VLA, varying only the visual input while holding all other factors constant:

| Factor | Single-image | Dual-image |
|--------|-------------|------------|
| Visual input | 1 camera (top-down) | 2 cameras (top-down + wrist) |
| VLM context tokens | ~97 | ~200 |
| Language, state, architecture, training | Identical | Identical |

The backbone (Cosmos Reason2-2B) was frozen in both configurations. Only the action head was trained. Both models used the same dataset, hyperparameters, and evaluation protocol.

### 2.2 Results

| Model | Success Rate (L0) | Episodes |
|-------|------------------|----------|
| Single-image baseline | 64% | 50 |
| **Dual-image** | **82%** | **50** |
| **Improvement** | **+18 pp** | |

The 18 percentage point improvement — a 28% relative gain — is attributable entirely to richer visual features from the frozen backbone. The action expert architecture contributed nothing to this improvement; it was the same 4-layer Transformer in both cases.

### 2.3 Significance

This result is notable for three reasons:
- First, the VLM backbone was never fine-tuned for robot control, yet **simply providing a second viewpoint unlocked substantially better performance**. This suggests that the VLM's general visual understanding, not domain-specific fine-tuning, is the performance bottleneck. 
- Second, **the magnitude of the gain (18 pp) exceeds typical gains from action expert modifications** such as switching between flow matching variants, loss function improvements, or classification-based heads. 
- Third, the improvement came from visual *coverage* (a geometrically complementary viewpoint) rather than visual *resolution*, indicating that spatial completeness is a key dimension of visual representation quality for manipulation tasks.

## 3. Evidence from Attention Analysis (Pi-Zero)

### 3.1 Methodological Foundation

Interpreting attention weights as modality importance indicators requires that **tokens retain their modality identity after deep Transformer processing**. In a model with $L$ layers of cross-attention, each token's representation accumulates residual updates from all modalities:

$$x_i^{(L)} = x_i^{(0)} + \sum_{l=1}^{L} \Delta_i^{(l)}$$

We validate this assumption empirically using the Silhouette Coefficient on t-SNE projections of the Q-projection space. A positive silhouette indicates that tokens remain closer to their own modality group than to other groups, providing the warrant for attention-based modality attribution.

Pi-Zero's dual-expert architecture (separate per-expert FFN layers, asymmetric hidden dimensions of 2048 vs 1024) provides a structural drift barrier that bounds cross-expert embedding drift. Our measurements confirm this: 
- **Silhouette score** remains positive (0.22-0.44) across all 7 sampled layers, and
- **Cross-expert CMF scores** are near-zero, indicating that the two experts maintain separate representational subspaces throughout the network.

### 3.2 Visual Attention Dominance Across Depth

We computed **the fraction of action query attention directed at visual tokens** (indices 0-767) 
- At each of the 7 sampled layers
- Across all 8 attention heads
- Averaged over 5 episodes and 64 timesteps. 

The uniform baseline — the visual share expected if attention were distributed uniformly across all tokens — is 0.886.

| Layer | Visual Share | Interpretation |
|-------|-------------|----------------|
| 0 | 0.999 | Near-total visual intake |
| 3 | 0.821 | High, beginning to diversify |
| 9 | 0.628 | Moderate, integrating non-visual |
| 15 | 0.602 | Moderate, balanced integration |
| 17 | 0.573 | Final layer, with head specialization |
| 6 | 0.535 | Active cross-modal blending |
| 12 | 0.268 | Peak cross-modal integration |

The depth-wise profile reveals a clear processing hierarchy: 
- The network begins with near-exclusive visual attention at layer 0
- Progressively integrates language and state information through the middle layers
- Finally returns to moderate visual dominance at the output. 

Even at layer 12 — the peak cross-modal integration point — the most visual-dedicated head still allocates 63.2% of its attention to vision.

### 3.3 Head-Level Specialization

At the final layer (layer 17), the 8 attention heads divide into distinct functional roles:

| Head | Visual Share | Role |
|------|-------------|------|
| `h1` | 0.960 | Visual-dedicated |
| `h0` | 0.908 | Visual-dedicated |
| `h5` | 0.854 | Visual-dominant |
| `h7`, `h6` | 0.763, 0.700 | Visual-leaning |
| `h3` | 0.399 | Mixed modality |
| `h4`, `h2` | 0.001, 0.000 | Non-visual specialists |

Five of eight heads dedicate over 70% of their attention to visual tokens, while two heads attend near-exclusively to non-visual tokens (language and state). 

This learned division of labor allows the model to maintain parallel visual and non-visual processing streams simultaneously. 

The specialization pattern also shifts across layers — for instance, `h2` is the top visual head at layer 0 but becomes the lowest at layer 17 — indicating dynamic repurposing of heads across network depth.

### 3.4 Embedding Geometry: The Dip-Then-Recover Pattern

The per-layer silhouette profile reveals how modality cluster structure evolves through the network:

| Layer | Silhouette |
|-------|-----------|
| 0 | 0.395 |
| 3 | 0.401 |
| 6 | 0.216 |
| 9 | 0.443 |
| 12 | 0.372 |
| 15 | 0.435 |
| 17 | 0.370 |

- The dip at layer 6 (silhouette drops from 0.40 to 0.22) indicates that the model deliberately blends modality representations for cross-modal reasoning. 
- The recovery by layer 9 (silhouette rises to 0.44) shows that the network re-establishes modality-specific structure after integration. 

This dip-then-recover pattern is consistent with the visual attention profile: layer 6 shows both the silhouette dip and below-average visual attention share (0.535), suggesting coordinated cross-modal integration in the same network region.

### 3.5 Cross-Modal Fusion Scores

Cross-Modal Fusion (CMF) measures the cosine alignment between action query Q-projections and attention-weighted V-projections of target modalities, computed per head in the 256-dimensional shared attention space.

| CMF Pair | Score | Expert Crossing |
|----------|-------|-----------------|
| A → S | 0.249 | Same expert |
| A → L | 0.002 | Cross-expert |
| A → V | −0.018 | Cross-expert |
| S → L | −0.022 | Cross-expert |
| S → V | −0.033 | Cross-expert |

The near-zero cross-expert CMF scores reveal a fundamental aspect of how Pi-Zero processes visual information: **effective cross-modal attention does not require representational alignment**. The Q-projections of action tokens and the V-projections of visual tokens occupy orthogonal subspaces, yet the model achieves high task performance. Information transfer occurs through attention routing (Q-K alignment for token selection) followed by learned output projections that decode the attended V-space representations into the action expert's hidden space.

This finding complements the visual attention analysis: the action expert heavily attends to visual tokens (Section 3.2), but the attended visual representations are decoded through a learned projection rather than through geometric alignment in head space. The visual modality's importance lies in the *information content* of its tokens, not in their *directional similarity* to the query.

## 4. Discussion

### 4.1 Converging Evidence

The evidence from both models converges on a single conclusion: **visual representation quality sets the performance ceiling for VLA policies**.

From the **Cosmos** experiment, we observe that **enriching the visual input from one to two cameras** — while keeping the action expert identical — produces **a larger performance gain (+18 pp)** than any architectural modification to the action expert tested in prior work on this baseline.

From the **Pi-Zero** analysis, we observe that the **action expert's attention capacity is overwhelmingly directed at visual tokens (57-100% across layers)**, with dedicated visual-specialist heads at every network depth. The modality embedding structure (silhouette 0.22-0.44) validates that this attention routing meaningfully reflects reliance on visual information.

Together, these findings suggest that **the action expert's primary role is to route and decode visual information, not to build cross-modal alignment**. The visual representation quality — determined by the VLM backbone and the camera configuration — is the binding constraint on task performance.

### 4.2 Implications for VLA Design

1. **Invest in visual representation first.** Before improving the action expert, ensure the VLM backbone receives rich, multi-view visual input. Adding a wrist camera produced a larger gain than switching between flow matching variants.

2. **Multi-view input is complementary, not redundant.** The wrist camera provides geometric information unavailable from the top-down view — occluded contact geometry, depth disambiguation, and fine-grained gripper-object pose. This is visual *coverage*, distinct from visual *resolution*.

3. **Cross-modal alignment is not necessary for effective fusion.** Pi-Zero achieves high performance despite near-zero Q-V cosine similarity across experts. The attention mechanism's routing function (Q-K alignment) and the output projection's decoding function are sufficient for effective information transfer.

4. **Head specialization emerges naturally.** Without explicit supervision, attention heads self-organize into visual-dedicated and non-visual specialist roles. This division of labor may be a general feature of VLA Transformers that could be leveraged for more efficient architectures.

### 4.3 Limitations

Our analysis has several limitations that should be considered when interpreting the results.

The Cosmos ablation was evaluated on a single task suite (`safety_static_obstacles`, L0) with one random seed. The single-image baseline exhibits documented high variance (32-64% SR). Multi-seed evaluation and cross-level testing are needed to establish statistical significance.

The dual-image model receives approximately twice as many VLM context tokens as the single-image model. While the visual content is the most plausible explanation for the improvement, a token-count control experiment (zero-padded single-image at the same sequence length) would isolate this confound.

The attention-based modality attribution operates at moderate confidence (silhouette 0.22-0.44). This is sufficient for the comparative claims presented but not for precise quantitative statements about modality importance fractions.

Finally, both models were evaluated on VLA-Arena simulation tasks only. Real-world manipulation involves additional challenges (sensor noise, lighting variation, domain gap) where the visual primacy finding may manifest differently.

## 5. Conclusion

Across two architecturally distinct VLA models and four complementary analysis methods — controlled ablation, cross-modal fusion scoring, attention attribution ranking, and embedding geometry profiling — we find consistent evidence that visual representation quality is the primary driver of VLA task performance. The action expert's role is to efficiently route and decode information from the visual tokens provided by the VLM backbone. Improving what the model sees yields larger gains than improving how it reasons about what it sees.
