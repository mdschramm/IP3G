"""
Early-training diagnostics for the conditional DDPM.

Run after 500-1000 local steps to validate model behaviour before committing
to a full remote run:

    conda run -n dataexplr python -m diffusion.diagnostics \
        --checkpoint output/diffusion/local/checkpoints/diffusion_model_step_000500.weights.h5 \
        --mode local
"""

import argparse
import csv
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from diffusion.diffusion_config import get_config
from diffusion.diffusion_model import build_unet
from diffusion.diffusion_train import WarmupCosineSchedule, WarmupFlatSchedule, get_learning_rate_schedule
import diffusion.diffusion_utils as diffusion_utils


# ─────────────────────────────────────────────────────────────────────────────
# 1. Learning-rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def check_lr_schedule(config):
    """Verify the LR schedule produces expected values."""
    kind = config.get('lr_schedule', 'cosine')
    s = get_learning_rate_schedule(
        config['learning_rate'], config['warmup_steps'], config['num_steps'],
        schedule_kind=kind,
    )
    w = config['warmup_steps']
    T = config['num_steps']

    key = [0, w // 4, w // 2, w, int(T * 0.25), int(T * 0.5), int(T * 0.75), T]
    print(f"\n── LR Schedule ({kind}) ──────────────────────────────────────────")
    for step in key:
        tag = "(warmup end)" if step == w else ""
        print(f"  step {step:6d}: {float(s(step)):.3e}  {tag}")

    assert abs(float(s(w)) - config['learning_rate']) < 1e-9, \
        f"LR at warmup end should equal base_lr, got {float(s(w)):.3e}"
    if kind == 'flat':
        assert abs(float(s(T)) - config['learning_rate']) < 1e-9, \
            f"Flat schedule: LR at end should equal base_lr, got {float(s(T)):.3e}"
        print("  ✓ Warmup peaks correctly; LR stays flat")
    else:
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
# 6. Magnitude history plotting
# ─────────────────────────────────────────────────────────────────────────────

def load_diag_history(path):
    """Load a magnitude diagnostic .npz (saved by _save_diag_history) into a nested dict."""
    data = np.load(path)
    result = {
        'steps': data['steps'],
        'weight_mean': {}, 'weight_max': {}, 'act_mean': {}, 'act_std': {},
        'gn_risk': {}, 'grad_norm': {}, 'grad_underflow': {}, 'scalar': {},
    }
    prefix_map = {
        'wmean':  'weight_mean',
        'wmax':   'weight_max',
        'amean':  'act_mean',
        'astd':   'act_std',
        'gnrisk': 'gn_risk',
        'gnorm':  'grad_norm',
        'guflow': 'grad_underflow',
        'scalar': 'scalar',
    }
    for key in data.files:
        if key == 'steps':
            continue
        if '__' not in key:
            continue  # skip keys without namespace separator rather than crashing
        prefix, name = key.split('__', 1)
        dest = prefix_map.get(prefix)
        if dest is not None:
            result[dest][name] = data[key]
    return result


def plot_magnitude_history(diag_file, output_path=None, top_n=8, weight_names=None, act_names=None):
    """Plot weight and activation magnitude trajectories from a diagnostic .npz file.

    By default selects the top_n layers by their final recorded magnitude. Pass
    weight_names / act_names to pin specific layers instead.

    Usage:
        # Top-8 by default
        python -m diffusion.diagnostics --plot-magnitudes path/to/magnitudes.npz

        # Top-12
        python -m diffusion.diagnostics --plot-magnitudes path/to/magnitudes.npz --top-n 12

        # Specific layers
        python -m diffusion.diagnostics --plot-magnitudes path/to/magnitudes.npz \
            --weight-names "res_net_block_0,res_net_block_1" \
            --act-names "res_net_block_2,res_net_block_5"

    Args:
        diag_file: Path to the .npz file produced during training.
        output_path: Where to save the PNG. Defaults to <diag_file>_plot.png.
        top_n: Number of layers to show when no explicit names are given.
            Selected by highest final magnitude value.
        weight_names: Comma-separated layer group names to plot, or a list of
            strings. Overrides top_n selection for weights.
        act_names: Comma-separated probe layer names to plot, or a list of
            strings. Overrides top_n selection for activations.

    Returns:
        output_path: Path where the plot was saved.
    """
    hist = load_diag_history(diag_file)
    steps = hist['steps']
    has_acts = bool(hist['act_mean'])

    # Accept comma-separated strings from CLI
    if isinstance(weight_names, str):
        weight_names = [n.strip() for n in weight_names.split(',') if n.strip()]
    if isinstance(act_names, str):
        act_names = [n.strip() for n in act_names.split(',') if n.strip()]

    def _top_keys(d, n, names):
        if names is not None:
            return [k for k in names if k in d]
        scored = {k: float(v[-1]) for k, v in d.items() if len(v) > 0}
        return sorted(scored, key=scored.get, reverse=True)[:n]

    w_keys = _top_keys(hist['weight_mean'], top_n, weight_names)
    a_keys = _top_keys(hist['act_mean'], top_n, act_names) if has_acts else []

    n_rows = 1 + int(bool(a_keys))
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows), squeeze=False)

    # Weight magnitudes
    ax = axes[0, 0]
    for layer_name in w_keys:
        ax.plot(steps, hist['weight_mean'][layer_name], label=layer_name, alpha=0.75)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean |w|')
    ax.set_title(f'Weight Magnitudes — top {len(w_keys)} by final magnitude')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Activation magnitudes
    if a_keys:
        ax = axes[1, 0]
        for layer_name in a_keys:
            ax.plot(steps, hist['act_mean'][layer_name], label=layer_name, alpha=0.75)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean |activation|')
        ax.set_title(f'Activation Magnitudes — top {len(a_keys)} by final magnitude')
        ax.legend(fontsize=8, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Magnitude History — {os.path.basename(diag_file)}', fontsize=11)
    plt.tight_layout()

    if output_path is None:
        output_path = diag_file.replace('.npz', '_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved magnitude plot to {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. CSV export of magnitude history
# ─────────────────────────────────────────────────────────────────────────────

def export_magnitude_csv(diag_file, output_path=None, top_n=15, sample_interval=500):
    """Export top-n weight and activation magnitudes to CSV.

    Rows are (step, kind, layer, value) in long/tidy format, sampled at the
    recorded step nearest to each sample_interval boundary.

    Usage:
        python -m diffusion.diagnostics --export-csv path/to/magnitudes.npz
        python -m diffusion.diagnostics --export-csv path/to/magnitudes.npz --top-n 20 --sample-interval 1000

    Args:
        diag_file: Path to the .npz file produced during training.
        output_path: Where to save the CSV. Defaults to <diag_file>_magnitudes.csv.
        top_n: Number of layers to include, ranked by final magnitude.
        sample_interval: Approximate step gap between rows (snaps to nearest recorded step).

    Returns:
        output_path: Path where the CSV was saved.
    """
    hist = load_diag_history(diag_file)
    steps = hist['steps']

    # Snap to the recorded step nearest each sample_interval boundary
    marks = np.arange(0, int(steps[-1]) + sample_interval, sample_interval)
    sampled_idx = sorted(set(int(np.argmin(np.abs(steps - m))) for m in marks))

    def _top_keys(d, n):
        if not d:
            return []
        scored = {k: float(v[-1]) for k, v in d.items() if len(v) > 0}
        return sorted(scored, key=scored.get, reverse=True)[:n]

    w_keys = _top_keys(hist['weight_mean'], top_n)
    a_keys = _top_keys(hist['act_mean'], top_n)

    if output_path is None:
        output_path = diag_file.replace('.npz', '_magnitudes.csv')

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'kind', 'layer', 'value'])
        for i in sampled_idx:
            step = int(steps[i])
            for k in w_keys:
                writer.writerow([step, 'weight_mean', k, f'{hist["weight_mean"][k][i]:.6g}'])
            for k in a_keys:
                writer.writerow([step, 'act_mean', k, f'{hist["act_mean"][k][i]:.6g}'])

    print(f"Saved magnitude CSV to {output_path}  "
          f"({len(sampled_idx)} steps × {len(w_keys)} weight + {len(a_keys)} activation layers)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Early-training DDPM diagnostics')
    parser.add_argument('--checkpoint', default=None,
                        help='Model checkpoint (.weights.h5) to analyse')
    parser.add_argument('--mode', choices=['local', 'remote', 'diagnostic'], default='local')
    parser.add_argument('--skip-grads', action='store_true',
                        help='Skip gradient check (faster, avoids Metal OOM on large model)')
    parser.add_argument('--plot-magnitudes', metavar='DIAG_NPZ', default=None,
                        help='Plot weight/activation magnitude history from a diagnostic .npz and exit')
    parser.add_argument('--top-n', type=int, default=8, metavar='N',
                        help='Number of layers to show, ranked by final magnitude (default: 8)')
    parser.add_argument('--weight-names', default=None, metavar='NAMES',
                        help='Comma-separated weight group names to plot instead of top-n')
    parser.add_argument('--act-names', default=None, metavar='NAMES',
                        help='Comma-separated activation layer names to plot instead of top-n')
    parser.add_argument('--export-csv', metavar='DIAG_NPZ', default=None,
                        help='Export top-n magnitude history to CSV and exit')
    parser.add_argument('--sample-interval', type=int, default=500, metavar='STEPS',
                        help='Step interval between CSV rows (default: 500)')
    args = parser.parse_args()

    if args.plot_magnitudes:
        plot_magnitude_history(
            args.plot_magnitudes,
            top_n=args.top_n,
            weight_names=args.weight_names,
            act_names=args.act_names,
        )
        sys.exit(0)

    if args.export_csv:
        export_magnitude_csv(
            args.export_csv,
            top_n=args.top_n,
            sample_interval=args.sample_interval,
        )
        sys.exit(0)

    if not args.checkpoint:
        parser.error('--checkpoint is required unless --plot-magnitudes or --export-csv is used')

    config = get_config(args.mode)
    X = np.load(f"output/preprocessing/{config['feature_file']}").astype(np.float32)
    y = np.load(f"output/preprocessing/{config['label_file']}").astype(np.float32)

    diffusion_utils.set_data_range(X.min(), X.max())

    model = build_unet(config)
    model.load_weights(args.checkpoint)
    print(f"Loaded {args.checkpoint}  ({model.count_params():,} params)")

    check_lr_schedule(config)
    check_timestep_loss(model, X, y, config)
    check_cfg_separation(model, X, y, config)
    if not args.skip_grads:
        check_gradient_norms(model, X, y, config)

    print("\n── Done ─────────────────────────────────────────────────────────\n")
