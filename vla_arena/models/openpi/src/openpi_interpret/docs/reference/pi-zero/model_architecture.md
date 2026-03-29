# Pi-Zero Model Architecture: A Beginner-Friendly Guide

This document explains the internal architecture of the **Pi-Zero (Pi0)** model as implemented in the OpenPI codebase. It is written for readers who are new to Pi-Zero and want to understand how visual observations, language instructions, robot state, and action trajectories flow through the model — from raw inputs to predicted actions.

> **Prerequisite reading**: [Training Pipeline Overview](training_pipeline_overview.md) covers data loading, transforms, and the training loop. This document focuses on what happens *inside* the model.

> **Source files**: All code references point to paths under  
> `vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi/models/`

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Four Types of Tokens](#2-four-types-of-tokens)
   - [Image Patch Tokens](#21-image-patch-tokens)
   - [Language Instruction Tokens](#22-language-instruction-tokens)
   - [Proprioceptive State Token](#23-proprioceptive-state-token)
   - [Timestamped Action Tokens](#24-timestamped-action-tokens)
3. [The Dual-Expert Backbone](#3-the-dual-expert-backbone)
   - [Why Two Experts?](#31-why-two-experts)
   - [How Cross-Expert Attention Works](#32-how-cross-expert-attention-works)
   - [The Prefix-LM Attention Mask](#33-the-prefix-lm-attention-mask)
   - [Per-Expert Processing Within Each Block](#34-per-expert-processing-within-each-block)
4. [Auxiliary Projectors](#4-auxiliary-projectors)
5. [The Flow Matching Head](#5-the-flow-matching-head)
   - [Training: Learning the Velocity Field](#51-training-learning-the-velocity-field)
   - [Inference: Iterative Denoising](#52-inference-iterative-denoising)
   - [What Connects the Backbone to the Flow Matching Head](#53-what-connects-the-backbone-to-the-flow-matching-head)
6. [Complete Forward Pass Walkthrough](#6-complete-forward-pass-walkthrough)
7. [Concrete Dimensions Reference](#7-concrete-dimensions-reference)

---

## 1. The Big Picture

Pi-Zero takes three kinds of sensory input — camera images, a language instruction, and the robot's current joint state — and produces a trajectory of future actions. The architecture has six stages:

```
 ┌────────────────────────── Prefix (Expert 0) ──────────────────────────┐
 │                                                                       │
 │  ┌──────────┐      ┌────────────┐                                     │
 │  │ SigLIP   │──▶   │            │                                     │
 │  │ ViT      │ 768  │            │                                     │
 │  │ (3 cams) │ tkns │            │                                     │
 │  └──────────┘      │   Gemma    │     ┌────────────────┐              │
 │                     │   Dual-    │────▶│ prefix_out     │ (discarded   │
 │  ┌──────────┐      │   Expert   │     │ [B,816,2048]   │  at infer)   │
 │  │PaliGemma │──▶   │   Backbone │     └────────────────┘              │
 │  │Embedder  │ 48   │   (18 lyrs)│                                     │
 │  └──────────┘ tkns │            │                                     │
 │                     │            │                                     │
 ├─────────────────────│            │────────────────────────────────────-┤
 │  Suffix (Expert 1)  │            │                                     │
 │                     │            │     ┌────────────────┐   ┌────────┐│
 │  ┌──────────┐      │            │────▶│ suffix_out     │──▶│action  ││
 │  │state_proj│──▶   │            │     │ [B, 51, 1024]  │   │out_proj││
 │  └──────────┘ 1tkn │            │     └────────────────┘   └───┬────┘│
 │                     │            │                              │     │
 │  ┌──────────┐      │            │                         v_t [B,50,32]
 │  │action +  │──▶   │            │                          (velocity) │
 │  │time MLPs │50tkns│            │                              │     │
 │  └──────────┘      └────────────┘                              ▼     │
 │                                                          Euler step   │
 │                                                          x_t += dt*v_t│
 └───────────────────────────────────────────────────────────────────────┘
```

The key architectural insight is: **all four modalities share a single attention computation** inside the Gemma backbone, but each modality's tokens are processed by different projection weights, allowing the backbone to act as both a vision-language encoder and an action decoder simultaneously.

---

## 2. Four Types of Tokens

Before entering the Gemma backbone, all inputs must be converted into token sequences. Pi-Zero constructs four distinct types of tokens, divides them into a **prefix** and a **suffix**, and feeds both through the backbone together.

### 2.1. Image Patch Tokens

**Source**: `siglip.py` (`_Module` class)  
**Produced by**: SigLIP ViT-So400m/14 vision encoder

Each camera image (224×224 RGB) is split into 14×14-pixel patches, producing a 16×16 grid = **256 patch tokens per camera**. Pi-Zero expects three camera views:

| Camera | Token count | Description |
|--------|------------|-------------|
| `base_0_rgb` | 256 | Third-person overview |
| `left_wrist_0_rgb` | 256 | Left wrist camera |
| `right_wrist_0_rgb` | 256 | Right wrist camera (may be masked) |

The SigLIP encoder (27-layer ViT, width 1152) processes each image independently and then projects each patch token to the PaliGemma expert's width (2048) via a learned linear head. The spatial structure is discarded after encoding — tokens are flattened into a 1D sequence — but the positional information is preserved through learned positional embeddings within SigLIP.

**Total image tokens**: up to 3 × 256 = **768 tokens**, each of dimension **2048**.

### 2.2. Language Instruction Tokens

**Source**: `gemma.py` (`Embedder` class)  
**Produced by**: PaliGemma's shared vocabulary embedder

The natural language instruction (e.g., *"pick up the red cup and place it on the shelf"*) is first tokenized into subword token IDs by PaliGemma's tokenizer (vocabulary size: 257,152). These IDs are then looked up in the `Embedder`'s learned embedding table and scaled by √width:

```python
x = self.input_embedding_table[(x,)]
x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
```

**Total language tokens**: up to **48 tokens** (padded to `max_token_len`), each of dimension **2048**.

### 2.3. Proprioceptive State Token

**Source**: `pi0.py` (`embed_suffix`)  
**Produced by**: `self.state_proj` (a learned linear projection)

The robot's current proprioceptive state — a 32-dimensional vector encoding joint angles, positions, and gripper state — is projected into the action expert's embedding space through a single linear layer:

```python
state_token = self.state_proj(obs.state)[:, None, :]  # [B, 32] → [B, 1, 1024]
```

This produces **1 token** of dimension **1024** (the action expert's width). The `[:, None, :]` adds a sequence dimension, turning the vector into a single-token sequence.

The state token serves as a bridge between the visual-language context (in the prefix) and the action trajectory (in the suffix). It tells the action expert *"where the robot is right now"*, grounding the action prediction in the current physical configuration.

### 2.4. Timestamped Action Tokens

**Source**: `pi0.py` (`embed_suffix`)  
**Produced by**: `self.action_in_proj` + `posemb_sincos` + `self.action_time_mlp_in/out`

Action tokens represent a trajectory of future robot actions being iteratively refined through the denoising process. Their construction has three sub-steps:

**Step 1 — Project noisy actions into embedding space:**

The current (noisy) action trajectory `x_t` has shape `[B, 50, 32]` — 50 future timesteps, each a 32-dimensional action. A linear projection maps each action step to the action expert's width:

```python
action_tokens = self.action_in_proj(noisy_actions)  # [B, 50, 32] → [B, 50, 1024]
```

**Step 2 — Encode the denoising timestep:**

The scalar denoising timestep `t ∈ (0, 1]` is encoded into a 1024-dimensional vector using sinusoidal positional encoding with periods tuned to the [0, 1] range:

```python
time_emb = posemb_sincos(timestep, 1024, min_period=4e-3, max_period=4.0)  # [B] → [B, 1024]
```

This embedding is then broadcast (replicated) to all 50 action positions:

```python
time_tokens = einops.repeat(time_emb, 'b emb -> b s emb', s=50)  # [B, 1024] → [B, 50, 1024]
```

**Step 3 — Fuse action and time information via MLP:**

The action embedding and time embedding are concatenated along the feature dimension and mixed through a 2-layer MLP with SiLU (swish) activation:

```python
action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)  # [B, 50, 2048]
action_time_tokens = self.action_time_mlp_in(action_time_tokens)             # [B, 50, 2048] → [B, 50, 1024]
action_time_tokens = nnx.swish(action_time_tokens)
action_time_tokens = self.action_time_mlp_out(action_time_tokens)            # [B, 50, 1024] → [B, 50, 1024]
```

The resulting **50 timestamped action tokens** (each 1024-dim) carry fused information about *"what noisy action is predicted at this future timestep"* and *"how far along the denoising process are we."* The timestep information is critical: it tells the model whether it should make large corrections (early in denoising, high `t`) or fine adjustments (late in denoising, low `t`).

### Summary: Token Inventory

| Token type | Count | Dimension | Expert | Sequence role |
|-----------|-------|-----------|--------|--------------|
| Image patches | 768 (3×256) | 2048 | Expert 0 (PaliGemma) | Prefix |
| Language tokens | ≤48 | 2048 | Expert 0 (PaliGemma) | Prefix |
| State token | 1 | 1024 | Expert 1 (Action) | Suffix |
| Action tokens | 50 | 1024 | Expert 1 (Action) | Suffix |
| **Total** | **~867** | | | |

---

## 3. The Dual-Expert Backbone

### 3.1. Why Two Experts?

A naive approach would be to use a single Transformer for everything. But vision-language understanding and action generation have very different computational demands:

- **Vision-language understanding** benefits from a large, pretrained model (PaliGemma, 2B parameters) that already knows how to ground language in images.
- **Action generation** needs to learn robot-specific dynamics from relatively small fine-tuning datasets — a smaller model (300M parameters) is sufficient and more parameter-efficient.

Pi-Zero's solution is a **dual-expert** Gemma backbone (`gemma.py`, `Module` class) that runs two experts **in parallel within the same Transformer stack**:

| | Expert 0 (PaliGemma) | Expert 1 (Action Expert) |
|---|---|---|
| **Role** | Vision-language context encoder | Action trajectory generator |
| **Width** | 2048 | 1024 |
| **MLP dim** | 16,384 | 4,096 |
| **Params** | ~2B | ~300M |
| **Processes** | Image patch + language tokens (prefix) | State + action tokens (suffix) |
| **Shared** | Depth (18 layers), num_heads (8), num_kv_heads (1), head_dim (256) |

The experts share the same **depth** (18 layers), **number of attention heads** (8 query heads, 1 KV head), and **head dimension** (256). They differ in **width** (hidden dimension) and **MLP dimension**. This shared head configuration is what makes cross-expert attention possible, as explained next.

### 3.2. How Cross-Expert Attention Works

This is the most subtle part of the architecture. The two experts have **different widths** (2048 vs. 1024), so their hidden states cannot be directly concatenated. The insight is that attention operates in **head space** (head_dim = 256), not in **model space** (width). The `Attention` class in `gemma.py` (lines 195–314) implements this in three phases:

**Phase 1 — Per-expert Q/K/V projection (different weights per expert):**

Each expert projects its tokens from its *own* width into the *shared* head space independently:

```
Expert 0: [B, 816, 2048] → Q₀[B, 816, 8, 256], K₀[B, 816, 1, 256], V₀[B, 816, 1, 256]
Expert 1: [B,  51, 1024] → Q₁[B,  51, 8, 256], K₁[B,  51, 1, 256], V₁[B,  51, 1, 256]
```

The projection matrices are *separate* for each expert (named with suffixes like `q_einsum` for Expert 0, `q_einsum_1` for Expert 1). This is critical — it means Expert 0's projection is initialized from the pretrained PaliGemma weights, while Expert 1's projection is trained from scratch.

**Phase 2 — Concatenate in head space and compute joint attention:**

After projection, Q, K, and V are **concatenated along the sequence axis**:

```python
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs))
# q: [B, 867, 8, 256]   (816 prefix + 51 suffix queries)
# k: [B, 867, 1, 256]   (816 prefix + 51 suffix keys)
# v: [B, 867, 1, 256]   (816 prefix + 51 suffix values)
```

Then **one unified softmax** is computed over the entire concatenated sequence:

```python
logits = jnp.einsum('BTKGH,BSKH->BKGTS', q, k)  # [B, 1, 8, 867, 867]
probs = jax.nn.softmax(masked_logits, axis=-1)     # [B, 1, 8, 867, 867]
encoded = jnp.einsum('BKGTS,BSKH->BTKGH', probs, v)
```

This is where the magic happens: **action tokens (Expert 1) can attend to image and language tokens (Expert 0)**, and vice versa (subject to the attention mask). The different expert widths don't matter because the attention computation happens entirely in the shared 256-dimensional head space.

**Phase 3 — Split and project back to per-expert widths:**

After attention, the output is split back by token ranges and projected to each expert's own width:

```python
# Expert 0 output: encoded[:, 0:816] projected via attn_vec_einsum [8, 256] → [2048]
# Expert 1 output: encoded[:, 816:867] projected via attn_vec_einsum_1 [8, 256] → [1024]
```

This three-phase design enables an elegant property: **each expert maintains its own representational capacity** (different widths) while **sharing attention patterns across all tokens** (same head space). The action expert can "read" the visual-language context without needing to match its dimensionality.

### 3.3. The Prefix-LM Attention Mask

Not all tokens should attend to all other tokens. Pi-Zero uses a **prefix-LM** attention pattern, constructed by the `make_attn_mask` function (`pi0.py`, lines 33–58):

```
                  ┌────── Prefix (816 tokens) ──────┐ ┌── Suffix (51 tokens) ──┐
                  │ img₀    img₁    img₂    lang    │ │ state  act₀ ... act₄₉  │
                  │ (256)   (256)   (256)   (≤48)   │ │  (1)   (1)  ...  (1)   │
 ─────────────────┼─────────────────────────────────┼─┼────────────────────────┤
 image₀ patches   │  ✓       ✓       ✓       ✓     │ │  ✗      ✗   ...   ✗    │
 image₁ patches   │  ✓       ✓       ✓       ✓     │ │  ✗      ✗   ...   ✗    │
 image₂ patches   │  ✓       ✓       ✓       ✓     │ │  ✗      ✗   ...   ✗    │
 language tokens   │  ✓       ✓       ✓       ✓     │ │  ✗      ✗   ...   ✗    │
 ─────────────────┼─────────────────────────────────┼─┼────────────────────────┤
 state token       │  ✓       ✓       ✓       ✓     │ │  ✓      ✗   ...   ✗    │
 action tokens     │  ✓       ✓       ✓       ✓     │ │  ✓      ✓   ...   ✓    │
```

The rules are:

1. **Prefix is bidirectional**: All image and language tokens attend freely to each other, enabling rich cross-modal visual-language understanding.
2. **Prefix cannot see suffix**: The visual-language context is computed independently of the action trajectory. This enables KV cache reuse during inference — the prefix is processed once and cached.
3. **Suffix sees everything**: Action tokens attend to the full prefix (all images and language) and to each other. This is how the action expert conditions its predictions on the visual-language context.
4. **State sees prefix, not actions**: The state token can read the visual-language context but forms a one-way boundary — it doesn't attend to action tokens.

This mask is implemented using a cumulative sum over boolean `ar_mask` flags. Tokens with `ar_mask=False` share the same attention block (bidirectional). Tokens with `ar_mask=True` start a new block that cannot be attended to by previous blocks.

### 3.4. Per-Expert Processing Within Each Block

Each of the 18 Transformer `Block`s (`gemma.py`, lines 350–422) processes both experts in parallel through four stages:

```
 xs = [prefix_tokens, suffix_tokens]    ← list of 2 tensors, different widths
          │
          ▼
 ┌─────────────────────────────────┐
 │ 1. Per-expert RMSNorm           │    Expert 0: RMSNorm over dim 2048
 │    (separate learned scales)    │    Expert 1: RMSNorm over dim 1024
 └──────────────┬──────────────────┘
                ▼
 ┌─────────────────────────────────┐
 │ 2. Joint Attention              │    Concatenate Q,K,V in head space,
 │    (shared softmax, per-expert  │    one unified softmax, split output,
 │     input/output projections)   │    per-expert output projection
 └──────────────┬──────────────────┘
                ▼
 ┌─────────────────────────────────┐
 │ 3. Per-expert RMSNorm           │    Same as step 1
 └──────────────┬──────────────────┘
                ▼
 ┌─────────────────────────────────┐
 │ 4. Per-expert Feed-Forward      │    Expert 0: MLP [2048 → 16384 → 2048]
 │    (separate weights, widths)   │    Expert 1: MLP [1024 →  4096 → 1024]
 └──────────────┬──────────────────┘
                ▼
 xs = [updated_prefix, updated_suffix]
```

Both RMSNorm and Feed-Forward are fully independent per expert (different learned parameters, different dimensions). Only the Attention step fuses information across experts.

---

## 4. Auxiliary Projectors

Pi-Zero uses five small auxiliary projection modules to bridge between the raw input/output spaces and the Transformer's embedding spaces. They are defined in the `Pi0.__init__` method (`pi0.py`, lines 87–153):

### `self.state_proj` — State Encoder

```
Linear(32 → 1024)
```

- **Purpose**: Projects the robot's 32-dim proprioceptive state into the action expert's embedding space.
- **Input**: `obs.state` — joint angles, gripper state.
- **Output**: A single 1024-dim token placed at the start of the suffix sequence.
- **Why it exists**: Raw proprioceptive values and Transformer embeddings live in completely different spaces. This learned projection maps physical quantities (angles in radians, gripper aperture) into the semantic space where the Transformer operates.

### `self.action_in_proj` — Action Encoder

```
Linear(32 → 1024)
```

- **Purpose**: Projects each 32-dim action step of the noisy trajectory into the action expert's embedding space.
- **Input**: `noisy_actions` of shape `[B, 50, 32]` — the current state of the action trajectory being denoised.
- **Output**: 50 tokens of dimension 1024.
- **Why it exists**: Same rationale as `state_proj`. Note that actions and state share the same dimensionality (32) but are projected by *separate* learned matrices, because they carry different semantics (state = "where am I", actions = "where should I go").

### `self.action_time_mlp_in` — Action–Time Fusion (Layer 1)

```
Linear(2048 → 1024)
```

- **Purpose**: First layer of a 2-layer MLP that fuses the action embedding (1024-dim) with the timestep embedding (1024-dim) after they are concatenated (2048-dim).
- **Input**: Concatenation of action tokens and broadcast timestep embedding.
- **Output**: Mixed 1024-dim representation, before activation.
- **Why it exists**: The denoising timestep tells the model how noisy the current trajectory is. Simply adding the time embedding to action tokens would not allow the model to learn complex interactions (e.g., "at high noise levels, rely more on the time signal; at low noise, rely more on the action signal"). The MLP enables non-linear mixing.

### `self.action_time_mlp_out` — Action–Time Fusion (Layer 2)

```
Linear(1024 → 1024)
```

- **Purpose**: Second layer of the fusion MLP, applied after swish activation.
- **Input**: 1024-dim intermediate from `action_time_mlp_in` after swish.
- **Output**: The final timestamped action tokens (1024-dim) ready for the Transformer.
- **Why it exists**: A single linear layer cannot learn complex fusion patterns. The two-layer MLP with non-linearity provides a universal function approximation capability for the fusion.

### `self.action_out_proj` — Action Decoder

```
Linear(1024 → 32)
```

- **Purpose**: Projects the Transformer's output embeddings back to the raw 32-dim action space.
- **Input**: The last 50 positions of the suffix output from the backbone (after all 18 layers).
- **Output**: `v_t` of shape `[B, 50, 32]` — the predicted velocity field for the flow matching update.
- **Why it exists**: The inverse of `action_in_proj`. Maps from the Transformer's semantic space back to physical action quantities that can be applied to the robot.

### Projector Data Flow Summary

```
                        ┌─────────────────────────────────┐
  obs.state ─────────►  │ state_proj  (32 → 1024)         │──► 1 suffix token
  [B, 32]               └─────────────────────────────────┘

                        ┌─────────────────────────────────┐
  noisy_actions ─────►  │ action_in_proj  (32 → 1024)     │──┐
  [B, 50, 32]           └─────────────────────────────────┘  │ concat along
                                                              ├──► features
  timestep ──► sincos ► │ repeat to [B, 50, 1024]         │──┘  [B, 50, 2048]
  [B]          [B,1024]  └─────────────────────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────┐
                        │ action_time_mlp_in (2048→1024)  │
                        │ swish                            │
                        │ action_time_mlp_out (1024→1024)  │──► 50 suffix tokens
                        └─────────────────────────────────┘    [B, 50, 1024]

                            ... Gemma backbone (18 layers) ...

                        ┌─────────────────────────────────┐
  suffix_out[:,-50:] ──►│ action_out_proj  (1024 → 32)    │──► v_t [B, 50, 32]
  [B, 50, 1024]         └─────────────────────────────────┘    (velocity field)
```

---

## 5. The Flow Matching Head

The flow matching head is not a separate neural network module — it is a *training and inference protocol* built on top of the dual-expert backbone and the `action_out_proj` projector. Understanding it requires grasping both the training objective and the inference procedure.

### 5.1. Training: Learning the Velocity Field

During training (`pi0.py`, `compute_loss`, lines 261–301), the model learns to predict the *direction* from clean actions to noise at any point along a linear interpolation path:

```
Given:
  actions    = ground-truth clean trajectory     [B, 50, 32]
  noise      = sampled from N(0, I)              [B, 50, 32]
  t          ~ Beta(1.5, 1) * 0.999 + 0.001     [B]  (scalar per sample)

Construct:
  x_t = t · noise + (1 - t) · actions           (noisy interpolation)
  u_t = noise - actions                          (target velocity: direction from clean → noise)

Forward pass:
  embed_prefix(observation)  →  prefix tokens
  embed_suffix(obs, x_t, t) →  suffix tokens    (x_t is the noisy trajectory)
  backbone([prefix, suffix]) →  [prefix_out, suffix_out]
  v_t = action_out_proj(suffix_out[:, -50:])     (predicted velocity)

Loss:
  MSE(v_t, u_t)                                  (match predicted vs. true velocity)
```

The Beta(1.5, 1) distribution for `t` slightly biases sampling toward higher noise levels, helping the model learn to make large corrections early in denoising.

Note that during training, prefix and suffix are processed in a **single joint forward pass** through the backbone (no KV caching). This is simpler and allows gradients to flow through the full sequence.

### 5.2. Inference: Iterative Denoising

During inference (`pi0.py`, `sample_actions`, lines 303–384), the model starts from pure noise and iteratively refines it into a clean action trajectory:

```
Initialize:
  x_1.0 = noise ~ N(0, I)           [B, 50, 32]
  dt = -1/num_steps                  (default: -0.1 for 10 steps)

Prefill (once):
  prefix_tokens = embed_prefix(observation)
  _, kv_cache = backbone([prefix_tokens, None])     ← cache prefix K,V

Denoise (10 iterations):
  for t = 1.0, 0.9, 0.8, ..., 0.1:
    suffix_tokens = embed_suffix(obs, x_t, t)
    _, suffix_out = backbone([None, suffix_tokens], kv_cache=kv_cache)
    v_t = action_out_proj(suffix_out[:, -50:])       ← predict velocity
    x_t = x_t + dt * v_t                             ← Euler step (dt < 0, moving toward clean)

Return x_0.0                                         ← denoised trajectory
```

Two important efficiency optimizations in the inference path:

1. **KV cache reuse**: The prefix (816 tokens) is processed once, and its Keys and Values are cached. Each denoising step only needs to forward the 51 suffix tokens against this cache, which is ~16× less compute per step.
2. **Suffix re-embedding**: The suffix is re-embedded from scratch at each denoising step because `x_t` changes. The timestep `t` also changes, so the timestamped action tokens are different every iteration.

### 5.3. What Connects the Backbone to the Flow Matching Head

The flow matching head's only learnable component is `action_out_proj` (a single linear layer). Everything else — the noise schedule, the Euler integration, the loss function — is pure math with no learned parameters.

What *drives* the iterative denoising process is the **information exchange inside the backbone**:

1. At each denoising step, the 50 action tokens enter the backbone carrying information about the *current* noisy trajectory and the *current* denoising timestep.
2. Through the 18 layers of cross-expert attention, action tokens attend to image patches (learning *what to interact with*), language tokens (learning *what task to perform*), and the state token (learning *where the robot currently is*).
3. The backbone's suffix output at the last 50 positions encodes a 1024-dimensional "action-contextualized" representation — this is the action expert's answer to *"given what I see, what I'm told, where I am, and how noisy the current trajectory is, which direction should I correct?"*
4. `action_out_proj` simply decodes this answer into a 32-dim velocity vector.
5. The Euler step applies the correction, and the process repeats with a cleaner trajectory.

The key conceptual point: **the backbone is not predicting actions directly**. It is predicting a *correction direction* (velocity field) that nudges the current noisy trajectory toward a valid action sequence. Over 10 small corrections, noise becomes a coherent, task-relevant trajectory.

---

## 6. Complete Forward Pass Walkthrough

Here is a concrete walkthrough of a single inference call for the VLA-Arena Pi-Zero model, tracing every tensor shape through the full pipeline.

### Step 1: Input Preparation

```
Inputs:
  base_0_rgb:          [1, 224, 224, 3]   float32, in [-1, 1]
  left_wrist_0_rgb:    [1, 224, 224, 3]   float32, in [-1, 1]
  right_wrist_0_rgb:   [1, 224, 224, 3]   float32, masked (zeros)
  state:               [1, 32]            float32
  tokenized_prompt:    [1, 48]            int32
  tokenized_prompt_mask: [1, 48]          bool
```

### Step 2: Prefix Construction (`embed_prefix`)

```
SigLIP(base_0_rgb)        → [1, 256, 2048]   (256 patch tokens)
SigLIP(left_wrist_0_rgb)  → [1, 256, 2048]   (256 patch tokens)
SigLIP(right_wrist_0_rgb) → [1, 256, 2048]   (256 patch tokens, masked out)
PaliGemma.embed(prompt)   → [1, 48, 2048]    (language tokens)

prefix_tokens  = concat → [1, 816, 2048]
prefix_mask    = concat → [1, 816]           (bool, right_wrist masked)
prefix_ar_mask = [False × 816]               (all bidirectional)
```

### Step 3: Prefix KV Cache (inference only)

```
backbone([prefix_tokens, None], mask=prefix_attn_mask)
  → kv_cache: K,V at each of 18 layers, shape [18, 1, 816, 1, 256] each
```

### Step 4: Suffix Construction (`embed_suffix`, per denoising step)

```
noise           = randn([1, 50, 32])          (at t=1.0)
state_proj(state) → [1, 1, 1024]              (1 state token)

action_in_proj(x_t)     → [1, 50, 1024]       (50 action tokens)
posemb_sincos(t=1.0)    → [1, 1024]           (timestep embedding)
repeat to               → [1, 50, 1024]       (broadcast to all 50 positions)
concat                  → [1, 50, 2048]       (action ⊕ time features)
action_time_mlp_in      → [1, 50, 1024]
swish
action_time_mlp_out     → [1, 50, 1024]       (50 timestamped action tokens)

suffix_tokens  = concat → [1, 51, 1024]       (state + 50 action tokens)
suffix_ar_mask = [True, True, False × 49]
```

### Step 5: Backbone Forward (per denoising step)

```
backbone([None, suffix_tokens], kv_cache=kv_cache, mask=full_attn_mask)

  Per layer (×18):
    RMSNorm(suffix_tokens)  → [1, 51, 1024]
    Q₁ = q_proj_1(normed)   → [1, 51, 8, 256]     ← suffix queries
    K₁ = k_proj_1(normed)   → [1, 51, 1, 256]     ← suffix keys
    V₁ = v_proj_1(normed)   → [1, 51, 1, 256]     ← suffix values

    K = concat(cached_K₀, K₁) → [1, 867, 1, 256]  ← all keys
    V = concat(cached_V₀, V₁) → [1, 867, 1, 256]  ← all values

    attention(Q₁, K, V, mask)  → [1, 51, 8, 256]   ← suffix attends to everything
    out_proj_1(attended)       → [1, 51, 1024]      ← back to suffix width
    residual + RMSNorm + FFN₁ + residual

  → suffix_out: [1, 51, 1024]
```

### Step 6: Velocity Prediction and Euler Step

```
v_t = action_out_proj(suffix_out[:, -50:])  → [1, 50, 32]   (velocity field)
x_t = x_t + (-0.1) * v_t                    → [1, 50, 32]   (one Euler step toward clean)
```

Steps 4–6 repeat 10 times (t = 1.0, 0.9, ..., 0.1). The final `x_0` is the predicted action trajectory.

---

## 7. Concrete Dimensions Reference

### VLA-Arena Configuration (`pi0_vla_arena_low_mem_finetune`)

| Parameter | Value | Source |
|-----------|-------|--------|
| `paligemma_variant` | `gemma_2b_lora` | `config.py` |
| `action_expert_variant` | `gemma_300m_lora` | `config.py` |
| `action_dim` | 32 | `pi0_config.py` default |
| `action_horizon` | 50 | `pi0_config.py` default |
| `max_token_len` | 48 | `pi0_config.py` default (Pi0) |
| `num_denoising_steps` | 10 | `sample_actions` default |

### Expert Configurations

| Parameter | Expert 0 (PaliGemma) | Expert 1 (Action) |
|-----------|---------------------|-------------------|
| Width (hidden dim) | 2048 | 1024 |
| MLP dim | 16,384 | 4,096 |
| Total params | ~2B | ~300M |
| Depth | 18 | 18 |
| Num heads (Q) | 8 | 8 |
| Num KV heads | 1 | 1 |
| Head dim | 256 | 256 |
| LoRA rank (attn) | 16 | 32 |
| LoRA rank (FFN) | 16 | 32 |

### Sequence Lengths

| Segment | Token count | Expert |
|---------|------------|--------|
| base_0_rgb patches | 256 | 0 |
| left_wrist_0_rgb patches | 256 | 0 |
| right_wrist_0_rgb patches | 256 | 0 |
| Language tokens | ≤48 | 0 |
| **Total prefix** | **≤816** | **0** |
| State token | 1 | 1 |
| Action tokens | 50 | 1 |
| **Total suffix** | **51** | **1** |
| **Grand total** | **≤867** | |

### Key Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `prefix_tokens` | `[B, 816, 2048]` | All image + language tokens |
| `suffix_tokens` | `[B, 51, 1024]` | State + timestamped action tokens |
| Q (per layer) | `[B, 867, 8, 256]` | Queries from both experts |
| K (per layer) | `[B, 867, 1, 256]` | Keys from both experts |
| `attn_mask` | `[B, 1, 867, 867]` | Prefix-LM mask |
| `suffix_out` | `[B, 51, 1024]` | Backbone output for suffix |
| `v_t` | `[B, 50, 32]` | Predicted velocity field |
| `x_0` (final) | `[B, 50, 32]` | Denoised action trajectory |
