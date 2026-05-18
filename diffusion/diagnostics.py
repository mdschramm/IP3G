"""
Early-training diagnostics for the conditional DDPM.

Run after 500-1000 local steps to validate model behaviour before committing
to a full remote run:

    conda run -n dataexplr python -m diffusion.diagnostics \
        --checkpoint output/diffusion/local/checkpoints/diffusion_model_step_000500.weights.h5 \
        --mode local
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from diffusion.diffusion_config import get_config
from diffusion.diffusion_model import build_unet
from diffusion.diffusion_train import WarmupCosineSchedule
import diffusion.diffusion_utils as diffusion_utils


# ─────────────────────────────────────────────────────────────────────────────
# 1. Learning-rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def check_lr_schedule(config):
    """Verify the warmup/cosine schedule produces expected values."""
    s = WarmupCosineSchedule(config['learning_rate'], config['warmup_steps'], config['num_steps'])
    w = config['warmup_steps']
    T = config['num_steps']

    key = [0, w // 4, w // 2, w, int(T * 0.25), int(T * 0.5), int(T * 0.75), T]
    print("\n── LR Schedule ──────────────────────────────────────────────────")
    for step in key:
        tag = f"(warmup end)" if step == w else ""
        print(f"  step {step:6d}: {float(s(step)):.3e}  {tag}")

    assert abs(float(s(w)) - config['learning_rate']) < 1e-9, \
        f"LR at warmup end should equal base_lr, got {float(s(w)):.3e}"
    assert float(s(T)) < config['learning_rate'] * 0.01, \
        f"LR at end should be ~0, got {float(s(T)):.3e}"
    print("  ✓ Warmup peaks correctly; cosine decays to near-zero")



# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-timestep loss breakdown
# ─────────────────────────────────────────────────────────────────────────────

def check_timestep_loss(model, X, y, config, n_batches=60):
    """Stratified loss by noise level — reveals where the model struggles."""
    diffusion_utils.init_schedule(config['timesteps'], kind=config['variance_schedule'])
    ab_all = diffusion_utils.alpha_cumprod
    rng = np.random.default_rng(42)

    buckets = {'t <200': [], '200–500': [], '500–800': [], 't >800': []}
    for _ in range(n_batches):
        idx = rng.integers(0, len(X), config['batch_size'])
        x0  = diffusion_utils.forward_transform(X[idx]).numpy()[..., np.newaxis]
        t   = rng.integers(1, config['timesteps'] + 1, config['batch_size'])
        eps = rng.standard_normal(x0.shape).astype(np.float32)
        ab  = ab_all[t][:, None, None, None]
        x_n = (np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps).astype(np.float32)

        pred        = model([x_n, t.astype(np.int32),
                             np.argmax(y[idx], axis=1).astype(np.int32)], training=False)
        per_sample  = np.mean((pred.numpy() - eps) ** 2, axis=(1, 2, 3))

        for ti, li in zip(t, per_sample):
            if   ti < 200: buckets['t <200'].append(li)
            elif ti < 500: buckets['200–500'].append(li)
            elif ti < 800: buckets['500–800'].append(li)
            else:          buckets['t >800'].append(li)

    print("\n── Per-Timestep Loss ────────────────────────────────────────────")
    means = []
    for k, v in buckets.items():
        m = np.mean(v)
        means.append(m)
        print(f"  {k:9s}: {m:.4f} ± {np.std(v):.4f}  (n={len(v)})")

    ratio = means[-1] / max(means[0], 1e-6)
    print(f"  t>800 / t<200 ratio: {ratio:.2f}x")
    if means[0] < means[-1]:
        print("  ✓ Low-noise loss < high-noise loss — directional structure learned")
    else:
        print("  ✗ Low-noise loss ≥ high-noise loss — model not yet predicting structure")
    if ratio < 1.5:
        print("  ⚠  Ratio <1.5× — loss is unusually flat across timesteps")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Classifier-free guidance separation
# ─────────────────────────────────────────────────────────────────────────────

def check_cfg_separation(model, X, y, config, n_classes=10, t_probe=400):
    """
    Measure how much class conditioning changes noise predictions vs unconditional.

    At t_probe (mid-noise), the crescent shape is still partially visible so
    class-specific guidance should be non-trivial if conditioning is working.
    """
    diffusion_utils.init_schedule(config['timesteps'], kind=config['variance_schedule'])
    ab_all = diffusion_utils.alpha_cumprod

    rng = np.random.default_rng(0)
    B = config['batch_size']
    idx = rng.integers(0, len(X), B)
    x0  = diffusion_utils.forward_transform(X[idx]).numpy()[..., np.newaxis]
    t   = np.full(B, t_probe, dtype=np.int32)
    eps = rng.standard_normal(x0.shape).astype(np.float32)
    ab  = ab_all[t][:, None, None, None]
    x_n = (np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps).astype(np.float32)

    uncond = np.full(B, config['num_classes'], dtype=np.int32)
    eps_u  = model([x_n, t, uncond], training=False).numpy()

    print(f"\n── CFG Separation (t={t_probe}, first {n_classes} classes) ───────────────")
    separations = []
    for c in range(min(n_classes, config['num_classes'])):
        cond   = np.full(B, c, dtype=np.int32)
        eps_c  = model([x_n, t, cond], training=False).numpy()
        l2     = float(np.sqrt(np.mean((eps_c - eps_u) ** 2)))
        separations.append(l2)
        print(f"  Class {c:2d}: ||eps_cond - eps_uncond||₂ = {l2:.4f}")

    mean_sep = np.mean(separations)
    spread   = np.std(separations)
    print(f"  Mean separation: {mean_sep:.4f}   class spread: ±{spread:.4f}")

    if mean_sep < 0.005:
        print("  ✗ Near-zero — class conditioning has no effect yet")
    elif spread / max(mean_sep, 1e-6) < 0.05:
        print("  ⚠  Low spread — conditioning active but classes look nearly identical")
    else:
        print("  ✓ Class-conditional predictions differ meaningfully across classes")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gradient norms by component
# ─────────────────────────────────────────────────────────────────────────────

def check_gradient_norms(model, X, y, config):
    """
    Single backward pass at mid-noise timesteps.
    Flags dead embedding gradients (class conditioning not learning) and
    imbalanced encoder/decoder norms.
    """
    diffusion_utils.init_schedule(config['timesteps'], kind=config['variance_schedule'])
    ab_all = diffusion_utils.alpha_cumprod

    rng = np.random.default_rng(1)
    B   = config['batch_size']
    idx = rng.integers(0, len(X), B)
    x0  = diffusion_utils.forward_transform(X[idx]).numpy()[..., np.newaxis]
    t   = rng.integers(200, 600, B).astype(np.int32)       # mid-noise
    eps = rng.standard_normal(x0.shape).astype(np.float32)
    ab  = ab_all[t][:, None, None, None]
    x_n = (np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps).astype(np.float32)
    cl  = np.argmax(y[idx], axis=1).astype(np.int32)

    with tf.GradientTape() as tape:
        pred = model([x_n, t, cl], training=True)
        loss = tf.reduce_mean(tf.square(pred - tf.cast(eps, pred.dtype)))
    grads = tape.gradient(loss, model.trainable_weights)

    global_norm = float(tf.linalg.global_norm([g for g in grads if g is not None]))

    groups   = {'embedding': [], 'attention': [], 'conv/resnet': [], 'norm': [], 'other': []}
    none_cnt = 0
    for w, g in zip(model.trainable_weights, grads):
        if g is None:
            none_cnt += 1
            continue
        norm = float(tf.norm(g))
        p    = w.path.lower()
        if 'embedding' in p or 'class' in p or 'time' in p:
            groups['embedding'].append(norm)
        elif 'attention' in p or 'einsum' in p:
            groups['attention'].append(norm)
        elif 'conv' in p or 'res_net' in p:
            groups['conv/resnet'].append(norm)
        elif 'norm' in p:
            groups['norm'].append(norm)
        else:
            groups['other'].append(norm)

    print(f"\n── Gradient Norms (mid-noise t=200–600) ─────────────────────────")
    print(f"  Global L2 norm: {global_norm:.4f}")
    for name, vals in groups.items():
        if vals:
            print(f"  {name:12s}: mean = {np.mean(vals):.4f}   "
                  f"max = {np.max(vals):.4f}   ({len(vals)} tensors)")
    if none_cnt:
        print(f"  ✗ {none_cnt} weight tensors received None gradient")

    emb_mean = np.mean(groups['embedding']) if groups['embedding'] else 0
    net_mean  = np.mean(groups['conv/resnet'] + groups['attention'] + groups['norm']) \
                if (groups['conv/resnet'] or groups['attention']) else 0

    if emb_mean < 1e-5:
        print("  ✗ Embedding gradients near-zero — class/time conditioning not learning")
    elif emb_mean < net_mean * 0.01:
        print("  ⚠  Embedding grads much smaller than network — conditioning may be slow")
    else:
        print("  ✓ Embedding gradients non-zero")

    if global_norm > 50:
        print(f"  ✗ Very high global norm ({global_norm:.1f}) — consider lower LR or tighter clip")
    elif global_norm < 0.01:
        print(f"  ✗ Very low global norm ({global_norm:.4f}) — model may have stalled")
    else:
        print(f"  ✓ Global gradient norm in plausible range")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Early-training DDPM diagnostics')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=['local', 'remote', 'diagnostic'], default='local')
    parser.add_argument('--skip-grads', action='store_true',
                        help='Skip gradient check (faster, avoids Metal OOM on large model)')
    args = parser.parse_args()

    config = get_config(args.mode)
    X = np.load(f"output/preprocessing/{config['feature_file']}").astype(np.float32)
    y = np.load(f"output/preprocessing/{config['label_file']}").astype(np.float32)

    diffusion_utils.set_data_range(X.min(), X.max())
    diffusion_utils.configure_log_transform(X, enable=config.get('log_transform', False))

    model = build_unet(config)
    model.load_weights(args.checkpoint)
    print(f"Loaded {args.checkpoint}  ({model.count_params():,} params)")

    check_lr_schedule(config)
    check_timestep_loss(model, X, y, config)
    check_cfg_separation(model, X, y, config)
    if not args.skip_grads:
        check_gradient_norms(model, X, y, config)

    print("\n── Done ─────────────────────────────────────────────────────────\n")
