# Diffusion Model Architecture

Conditional image generation on 128×128 GTEx gene expression images (16 channels by
default; resolution and channel count are both configurable — see
`preprocessing/artifact_paths.py`).
54 tissue/disease classes + 1 unconditional token for classifier-free guidance.
Architecture family: EDM2 (Karras et al. 2023) U-Net with magnitude-preserving (MP) layers.

---

## 1. EDM Preconditioning and Loss Target

*Source: `diffusion_utils.py:prepare_batch_conditional_edm`, `diffusion_train.py:train_step`*

### Forward process

Additive noise — not the DDPM multiplicative mixture:

```
x_t = x_0 + σ·ε,    ε ~ N(0, I)
```

At training, σ is drawn from a truncated log-normal:

```
ln(σ) ~ N(P_mean=-1.77, P_std=1.2),   σ ∈ [0.002, 80.0]
```

`exp(P_mean) ≈ 0.171 ≈ σ_data`, so the distribution is centred near the data scale,
concentrating gradient signal where the denoiser is most uncertain. (Already re-centred
from the EDM2 paper default P_mean=-2.0 to match this dataset's measured σ_data — see
§8 item 7.)

### Preconditioning scalars

All are deterministic functions of σ and the measured data standard deviation
σ_data = 0.1709 (128×128×16 config; recompute from `sigma_data.json` if width, height,
or channel count change):

| Scalar | Formula | Purpose |
|--------|---------|---------|
| `c_in` | 1/√(σ²+σ_data²) | Scale model input to unit RMS |
| `c_skip` | σ_data²/(σ²+σ_data²) | Skip weight → 1 at low σ (nearly clean input) |
| `c_out` | σ·σ_data/√(σ²+σ_data²) | Residual weight → 0 at low σ |
| `c_noise` | ln(σ)/4 | Compressed time signal passed to the sinusoidal embedding |

### Denoised estimate

The model `F_θ` does not directly predict the denoised image. The denoised estimate is:

```
D(x_t, σ, c) = c_skip · x_t  +  c_out · F_θ(c_in · x_t,  c_noise,  c)
```

At low σ, c_skip → 1 and c_out → 0, so D ≈ x_t (the nearly-clean input passes through).
At high σ, c_skip → 0 and c_out → σ_data, so D ≈ σ_data · F_θ (the model carries everything).

### Training target for F_θ

Solving D = x_0 for F_θ:

```
F_target = (x_0  −  c_skip · x_t) / c_out
```

This parameterisation makes the effective per-sample loss weight `w(σ) · c_out²(σ) = 1`
at every noise level, so plain MSE on `F_θ` is correctly weighted without any
explicit σ-dependent multiplier.

### Loss (EDM2 adaptive uncertainty)

```
L = E[ ‖F_θ − F_target‖²  / exp(logvar)  +  logvar ]
```

`logvar` is a learned per-σ scalar produced by a separate head (see §3).
It is safety-clamped to [−10, 5] before use so a single bad batch cannot cause
irreversible divergence. The upper bound is tighter because large positive logvar
collapses the loss to a constant independent of the prediction.

---

## 2. Conditioning

*Source: `diffusion_model.py:TimeAndClassEmbedding`, `AdaGN`*

### Conditioning vector construction

```
c_noise  →  get_sinusoidal_embeddings(c_noise, emb_dim)
                      ↓
          MPLinear(emb_dim×4) → MPSiLU → MPLinear(emb_dim)    [time_mlp]
                      ⊕
class_label  →  Embedding(num_classes+1, emb_dim)              [+1 = unconditional token]
                      ↓
          conditioning  (B, emb_dim)  — broadcast into every ResNetBlock
```

The unconditional token is class index 54 (= `num_classes`). During training, 10% of
labels are replaced by this token (classifier-free guidance dropout), enabling CFG at
inference without a separate unconditional model.

### AdaGN — Adaptive Group Normalization

Inside each ResNetBlock, features are modulated by the conditioning vector:

```
h = GroupNorm(x)                             # float32, avoids FP16 cancellation in (x−μ)/σ
[scale_raw, shift] = MLP(conditioning)       # zero-initialised weights (adaLN-Zero)
output = cast((1 + scale_raw) · h + shift, x.dtype)
```

**Zero initialisation** of the MLP kernel means `scale_raw = 0` at init, giving an identity
passthrough. Conditioning learns to deviate from identity only when the loss requires it,
preventing early instability (DiT adaLN-Zero pattern).

### Occupancy mask conditioning

A third conditioning signal, separate from the time/class vector above: the per-pixel
occupancy mask (1 where a gene actually maps to that pixel, 0 elsewhere — produced
alongside `resized_expressions.npy` by `preprocessing.prepare_training_data`). It's
projected into feature space and added right after the initial convolution:

```
h = MPConv2D(ch[0], 3)(x_noisy)
h = h + MPConv2D(ch[0], 1, use_bias=False)(occupancy_mask)
```

`use_bias=False` so unoccupied positions contribute exactly zero, not a learned offset —
the model can only receive positive evidence that a pixel is real, never a bias term that
would leak information into structurally-empty positions. See §3 for the matching
output-side enforcement and §6 for how the mask is used during sampling.

---

## 3. Network Architecture

*Source: `diffusion_model.py:build_unet`*

### Inputs and outputs

```
Inputs:  x_noisy        (B, 128, 128, 16)  float32  — c_in-scaled noisy image
         timesteps      (B,)               float32  — c_noise = ln(σ)/4
         class_labels   (B,)               int32    — tissue class or unconditional token
         occupancy_mask (B, 128, 128, 16)  float32  — 1 at real gene-pixel positions, 0 elsewhere

Outputs: F_pred         (B, 128, 128, 16)  — denoising residual (see §1); hard-masked to
                                              0 at unoccupied positions (see below)
         logvar         (B, 1, 1, 1)       — per-σ log variance (broadcasts over spatial dims)
```

Shapes shown are the current default (128×128×16); all four scale with whatever
`image_size`/`in_channels` the active config specifies.

### U-Net structure (remote/diagnostic config: channels = [32, 64, 128, 256])

```
x_noisy                                    occupancy_mask
  └─ MPConv2D(ch[0], 3)                          │
       └─ + MPConv2D(ch[0], 1, no bias) ◄─────────┘   [additive mask conditioning, §2]
       └─ Encoder level 0  (128×128, ch=32)
            num_res_blocks × ResNetBlock
            [SelfAttention if resolution in attention_resolutions]
            clip(h, ±act_clip)
            → skip_0
            └─ Downsample (stride-2 MPConv2D) → 64×64
       └─ Encoder level 1  (64×64, ch=64)   ... → skip_1 → 32×32
       └─ Encoder level 2  (32×32, ch=128)  ... → skip_2 → 16×16
       └─ Encoder level 3  (16×16, ch=256)  ... → skip_3  [no downsample at last level]
            └─ Bottleneck
                 ResNetBlock → SelfAttention → ResNetBlock
       └─ Logvar head (parallel, from c_noise only):
            MPFourier(128) → MPLinear(1) → Reshape(1,1,1)
       └─ Decoder level 3  (16×16)  cat(skip_3); ResNetBlocks; Attention; clip
       └─ Decoder level 2  (32×32)  Upsample; cat(skip_2); ResNetBlocks; clip
       └─ Decoder level 1  (64×64)  Upsample; cat(skip_1); ResNetBlocks; clip
       └─ Decoder level 0  (128×128) Upsample; cat(skip_0); ResNetBlocks; clip
            └─ GroupNorm(f32) → MPSiLU → MPConv2D(in_channels, 3) → ×occupancy_mask → F_pred
```

Attention resolutions in the remote/diagnostic config: {16} (bottleneck only).
`act_clip = 256.0` clamps encoder and decoder block outputs to prevent rare FP16 overflow
spikes from propagating through skip connections.

### Hard occupancy enforcement

The final output is multiplied by the occupancy mask before being returned:
`output = F_pred × occupancy_mask`. Paired with the masked loss in
`diffusion_train.py:train_step` (which sums the loss only over occupied positions), this
guarantees the model never predicts anything at structurally-unoccupied channels and is
never penalized for whatever it outputs there — the mask is enforced architecturally,
not just encouraged by the loss.

### Logvar head isolation

The logvar head branches from `c_noise` (the time input), **not** from the spatial bottleneck
features. This is critical: logvar should encode uncertainty as a function of noise level σ
alone. Branching from the decoder entangles it with spatial content, causing Adam to overshoot
logvar negative near peak LR and driving `MSE / exp(logvar)` → 1e8.

---

## 4. Magnitude-Preserving (MP) Design

*Source: `diffusion_model.py:MPConv2D`, `MPLinear`, `MPSiLU`, `ResNetBlock`, `SelfAttention`*

### Motivation

Without MP, conv/linear weight norms grow alongside activations during training.
Observed in diagnostic runs: mean activation 4.2 → 23.6, loss 0.09 → 0.248 at step 1600,
root-caused to unconstrained weight growth amplifying features through each layer.

### MPConv2D / MPLinear — on-the-fly weight normalisation

```
ŵ_i = w_i / (‖w_i‖₂ + ε),    ε = 1e-4
```

Applied per output filter (column of the weight matrix). Weights are trained normally
via gradient descent; normalisation is applied only during the forward pass and is never
inverted. This matches EDM2 Config D (§B.4).

### MPSiLU — activation variance preservation

```
MPSiLU(x) = SiLU(x) / 0.596
```

0.596 = E[SiLU(x)²]^{1/2} for x ~ N(0,1). Dividing by this constant keeps output
variance ≈ input variance, completing the MP chain through non-linearities.

### ResNetBlock — asymmetric magnitude-preserving residual sum

```
output = ((1 − t) · x_skip  +  t · h_residual)  /  √((1−t)² + t²)
```

with `t = res_balance = 0.3`. The denominator normalises so that output variance equals
input variance for uncorrelated equal-magnitude branches. t = 0.3 gives a 70/30
skip/residual split (EDM2 default). This is more conservative than the symmetric t = 0.5
case and prevents early residual branches from dominating before they have been trained.

### SelfAttention — magnitude-preserving residual

```
output = (x  +  attention_output)  ×  2^{−0.5}
```

Division by √2 preserves variance when `x` and `attention_output` are uncorrelated and
of equal magnitude, which they are at initialisation (attention of GroupNorm output ≈ 1).

### QKNorm — attention logit stabilisation

After the MHA projections produce Q and K of shape (B, seq_len, heads, head_dim),
LayerNorm is applied along `axis=-1` (per head, per position):

```
Q̂ = LayerNorm(Q),    K̂ = LayerNorm(K)    [both float32]
scores = Q̂ · K̂ᵀ / √head_dim
```

Without this, attention logits scale as ‖W_q‖ · ‖W_k‖, which grows quadratically with
weight magnitude and collapses softmax to a hard argmax — freezing attention gradients.
QK-norm caps logit magnitude at ~√head_dim regardless of weight scale.
Reference: Zhai et al. 2022, "Scaling ViT to 22B Parameters."

---

## 5. Training

*Source: `diffusion_train.py`*

| Hyperparameter | Diagnostic config (remote, A100) | Local config (Mac) |
|---|---|---|
| Steps | 30,000 | 1,000 |
| Batch size | 32 | 8 |
| Optimizer | AdamW(β₁=0.9, β₂=0.99, wd=0.01, clipnorm=1.0) | same |
| LR schedule | warmup(1.5k) + cosine decay | warmup(50) + flat |
| Base LR | 3e-4 | 3e-4 |
| EMA decay | 0.9995 | 0.99 |
| Mixed precision | FP16 + LossScaleOptimizer | FP16 + LossScaleOptimizer — deliberately kept on (not disabled) so Metal-specific FP16 issues surface locally before they cost A100 hours |
| CFG dropout | 10% | 10% |
| Excluded classes | [6, 24, 25, 31] (< 10 samples) | same |

There is no separate "remote" config — `diagnostic` (full remote architecture, bounded step count) is what actually
runs on the A100; a `CONFIG_REMOTE` with an open-ended 500k-step schedule existed at one point but was unused/stale
and has been removed from `diffusion_config.py`.

**EMA** shadow weights are maintained in float32. The update `(1 − decay) ≈ 1e-4`
underflows in FP16, which would cause the shadow to never move. EMA weights are
swapped in for sample generation intervals and restored afterwards.

**Diagnostics** (configurable interval, default 500 steps for diagnostic mode):
- Per-group mean/max weight magnitude and inter-step weight delta
- Per-layer mean/std activation magnitude and FP16 GroupNorm precision risk ratio
- Per-group gradient L2 norm and FP16 underflow fraction
- Loss scale value (from LossScaleOptimizer)
- AdaGN scale_raw magnitude (healthy ≈ 0; growing past 2 warrants attention)

---

## 6. Sampling — EDM Heun ODE

*Source: `diffusion_edm_sample.py`*

### σ schedule (Karras et al. 2022 §5)

```
σ_i = (σ_max^{1/ρ}  +  i/(N−1) · (σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ,    i = 0…N−1
```

with ρ = 7, σ_max = 80, σ_min = 0.002, N = 40 (inference default). The schedule is
appended with 0 for the final clean-image step. ρ = 7 allocates more steps near σ_min
where fine-grained details emerge and the model is most sensitive to step size.

### Initial sample

```
x ~ N(0, σ_max² · I)
```

At σ = 80 >> σ_data = 0.1709, the data signal is negligible: x_t ≈ σ·ε, so this is the
correct marginal distribution of the forward process. After scaling: `c_in · x ~ N(0, I)`.

### Occupancy masking during sampling

The initial noise, and the state after every Euler/Heun step, is masked to the same
occupancy pattern used in training: `x = x * occupancy_mask` (`sample_edm_batch` in
`diffusion_edm_sample.py`). This keeps structural zeros exactly zero throughout the ODE
trajectory rather than relying on the network to learn them from noise at every step. If
no mask is supplied, sampling falls back to a dense all-ones mask (backward-compatible
with single-channel, unmasked checkpoints).

### Per-step denoiser (`_denoise`)

1. Compute c_skip, c_out, c_in, c_noise from σ (identical formulae to training)
2. Scale input: `x_in = c_in · x`  (unit RMS for the model)
3. **Concatenated CFG forward pass** — cond and uncond in a single batch of size 2N:

```
[F_uncond; F_cond] = F_θ([x_in; x_in], [c_noise; c_noise], [uncond_labels; cond_labels])
F_cfg = F_uncond  +  w · (F_cond − F_uncond)
```

4. Denoised estimate: `D = c_skip · x  +  c_out · F_cfg`
5. ODE direction: `d = (x − D) / σ`

The logvar output is discarded at sampling time (`F_both, _ = model(...)`).

### Heun 2nd-order integration (Algorithm 1, Karras 2022)

```
for (σ_i, σ_{i+1}) in zip(sigmas[:-1], sigmas[1:]):

    d_i = (x − D(x, σ_i)) / σ_i                           # ODE direction at start of step
    x_euler = x + (σ_{i+1} − σ_i) · d_i                   # Euler predictor

    if σ_{i+1} > 0:
        d_next = (x_euler − D(x_euler, σ_{i+1})) / σ_{i+1}  # ODE direction at end of step
        x = x + (σ_{i+1} − σ_i) · (d_i + d_next) / 2     # Heun corrector (average slopes)
    else:
        x = x_euler                                         # Final step: corrector divides by 0
```

Heun achieves O(h²) local truncation error vs O(h) for Euler, so 40 Heun steps gives
quality comparable to ~200 Euler steps. The final step skips the corrector because
`d_next = (x − D) / 0` is undefined.

### Classifier-free guidance

CFG is applied to the raw model output F_θ before preconditioning reconstruction:

```
F_cfg = F_uncond  +  w · (F_cond − F_uncond)
D = c_skip · x  +  c_out · F_cfg
```

Applying CFG to F_θ (not to D directly) is consistent with training and avoids
amplifying the guidance signal through the c_out scaling factor.

---

## 7. Key Differences from the Original Papers

### vs. EDM (Karras et al. 2022, arXiv:2206.00364)

| Aspect | EDM paper | This implementation |
|--------|-----------|---------------------|
| Network | Standard U-Net + GroupNorm | MP-U-Net: MPConv2D, MPLinear, MPSiLU throughout |
| Loss weighting | Fixed `w(σ) = (σ²+σ_data²)/(σ·σ_data)²` via preconditioning | EDM2 adaptive logvar head; per-σ uncertainty is learned |
| Attention | Standard MHA | QKNorm MHA; prevents logit explosion from weight growth |
| Residual | Plain `x + h` | Asymmetric MP sum `((1−t)x + th) / ‖(1−t, t)‖`, t=0.3 |
| Conditioning | Unconditional or class via label embedding | AdaGN with zero-init MLP (adaLN-Zero); class + time fused |
| Precision | Float32 | Mixed FP16; GroupNorm, gradient penalty, logvar arithmetic stay FP32 |
| Sampling | 2nd-order Heun ODE | Same (Algorithm 1); concatenated CFG forward pass for efficiency |
| Logvar head | Not present (fixed loss weighting) | Separate MPFourier → MPLinear head isolated from spatial features |

### vs. EDM2 (Karras et al. 2023, arXiv:2312.02696)

| Aspect | EDM2 paper | This implementation |
|--------|------------|---------------------|
| Time embedding | MPFourier → main conditioning path | Sinusoidal embeddings → MPLinear MLP; MPFourier only for logvar head |
| Conditioning layer | Adaptive LayerNorm (AdaLN) | Adaptive GroupNorm (AdaGN) — better for spatial features at small channel counts |
| Data domain | Natural images (CIFAR-10, ImageNet) at multiple resolutions | Multichannel gene expression images (16-channel × 128×128 default); resolution and channel count are both config-driven |
| σ_data | ~0.5 (natural images) | 0.1709 (measured from GTEx dataset, 128×128×16 config — recompute per config via `sigma_data.json`) |
| Structural masking | Not present (dense images) | Occupancy mask conditions the input, hard-zeroes the output, and restricts the training loss to real (non-structural-zero) positions — needed because gene-expression images are majority empty (§2, §3) |
| Network scale | Hundreds of channels; very deep | channels=[32,64,128,256]; 3 ResNet blocks per level |
| Weight normalisation | Per-filter unit-norm, Config D, ε=1e-4 | Same (MPConv2D/MPLinear) |
| Logvar head | MPFourier → MPLinear scalar | Same design; branch isolation from spatial features is explicit |
| Sparse attention | Not present | SparseSelfAttention (top-k multiplicative gating) available for sparse gene-expression pixels |
| Unconditional token | Mapped null label | Class index 54 (= num_classes) |
| EMA | β=0.9999 | Same; shadow weights in float32 to avoid fp16 underflow of (1−β) = 1e-4 |
| CFG forward pass | Separate cond/uncond calls | Single concatenated 2N batch; halves kernel launches |

---

## 8. Suggested Next Steps

Ordered by where the project actually stands, not just each item's individual merit.
`build_unet()` did not construct successfully at *any* resolution until 2026-08-28:
`tf.cast` was applied directly to the `mask_input` KerasTensor placeholder outside any
layer's `call()`, which the installed Keras 3.10 rejects under the Functional API (fixed —
now `keras.ops.cast`, `diffusion_model.py:644,720`). Per the commit history, the masked
architecture has never completed a validated training run. Items below that assume a
working baseline are sequenced after that prerequisite, not before it.

### Do first

**0. Complete one successful training run on the current masked architecture.**
Confirm loss decreases, the EMA/mixed-precision/AdaGN diagnostics (§5) stay in their
healthy ranges, and a sampled image round-trips sensibly through the occupancy mask
(§2, §3, §6). This is a prerequisite for most of Tier 2 and all of Tier 3 below —
items 3, 4, 6, 8, 9, and 10 each need either a validated baseline to compare against
or a trained model to generate/distil from — not an independent item on the list.

### Tier 1 — safe, cheap, valuable regardless of what else changes

**1. EMA of Adam second moments** *(training)*
Currently only weight EMA is tracked. Adam's second moment accumulates from scratch
after every resume, causing a warm-up period with effectively too-large step sizes.
Saving and restoring optimizer state alongside the EMA checkpoint removes this —
contained, low-risk, and a direct payoff for a project that resumes/restarts training
frequently (see commit history).

**2. Class-frequency-weighted CFG dropout** *(training)*
`excluded_classes: [6, 24, 25, 31]` only removes the n<10 tail; classes with n=28–60
(e.g. class 34, n=28) remain in training under a flat 10% unconditional dropout rate,
which may be too aggressive for them specifically. A per-class rate proportional to
1/√n would protect rare-class CFG signal without touching common classes. Cheap,
well-scoped, and directly testable once a baseline run exists to compare against.

### Tier 2 — real value, sequence after a validated baseline

**3. Channel attention at bottleneck** *(architecture)*
Add squeeze-excitation or channel-MHA at the 16×16 bottleneck to capture cross-gene
correlations that spatial MHA misses. More motivated now than when originally proposed:
a correlation check on the 16-channel scheme's overflow channel found its collapsed
(averaged) genes are barely correlated with each other (median r≈0.05, close to the
random-gene-pair baseline of ~0.07) — a channel-attention mechanism is one plausible way
to let the model exploit gene co-expression structure that pixel-averaging currently
just discards.

**4. FID / KID against held-out real images** *(evaluation)*
The current evaluation pipeline (classifier confidence / class fidelity) measures how
well generated images are classified, not how realistic they look as distributions.
Literal Inception-based FID doesn't transfer to this domain — there's no meaningful
gene-expression feature extractor pretrained elsewhere. Adapt it instead: use the
existing trained tissue classifier's penultimate-layer activations as the feature space
for a Fréchet distance.

**5. MPFourier for main time conditioning** *(architecture)*
Replace `get_sinusoidal_embeddings` with `MPFourier` (already implemented for the logvar
head) in the main `TimeAndClassEmbedding` path. Completes the MP chain end-to-end and
matches EDM2 Config D exactly. Contained to `TimeAndClassEmbedding.build`; safe to bundle
into a later cleanup pass rather than treat as urgent.

### Tier 3 — defer

**6. AdaGN → Adaptive LayerNorm (AdaLN)** *(architecture)*
EDM2 recommends AdaLN over AdaGN — it normalises over all channels rather than groups,
removing the float32 GroupNorm casting overhead. Real risk: the codebase carries visible
scar tissue from past instability (§4 Motivation, QKNorm, act_clip, adaLN-Zero, fp32 EMA),
all tuned around AdaGN's current behaviour, and `res_balance` would likely need retuning
since the norm statistics change. Treat as a controlled experiment against a working
baseline, not a default swap.

**7. Re-center P_mean / P_std for GTEx** *(training)*
Partially done already: `diffusion_config.py` now sets `P_mean = -1.77`
(≈ ln(σ_data = 0.1709), not the EDM2 paper default of -2.0 — see §1). The remaining,
more sophisticated piece — fitting P_mean to the peak of the empirical
denoiser-MSE-vs-σ curve rather than just matching σ_data — is a refinement, not a
correctness fix, and premature while resolution/channel config may still change
(σ_data measured 0.1586–0.1603 for 256²/512² single-channel candidates evaluated
separately — close enough that neither choice needs dramatic rework).

**8. Evaluate stochastic SDE sampling** *(sampling)*
The deterministic ODE sampler (η=0) can accumulate early denoising errors; the EDM SDE
variant (η>0) injects noise per step and can recover from them. Plausibly useful for
reducing residual background fog in sparse images specifically — but is a sampling-time
optimisation for a model that doesn't yet have a completed training run to sample from.

**9. Consistency distillation** *(sampling)*
Song et al. 2023: an EDM-trained model can be distilled into a 1–4 step consistency
model (~20× faster than the current 40-step/80-eval Heun sampler). Requires a trained
model to distil from first — correctly sequenced last regardless of individual merit.

**10. MP ablation** *(evaluation)*
Retrain with standard Conv2D/Linear instead of MPConv2D/MPLinear to quantify their
contribution on this dataset specifically. Useful for a writeup, not for engineering
decisions — the documented instability incidents (§4) already constitute strong evidence
MP matters; a controlled ablation is confirmatory, not decision-blocking, and costs a
full retrain to run.
