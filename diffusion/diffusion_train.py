"""
Training script for Conditional DDPM with Classifier-Free Guidance.

Implements:
- Training loop with EMA (Exponential Moving Average)
- Checkpointing
- Sample generation during training
- Loss tracking and visualization
- Learning rate scheduling
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from datetime import datetime

from diffusion.diffusion_config import get_config, print_config
from diffusion.diffusion_model import build_unet
import diffusion.diffusion_utils as diffusion_utils
from preprocessing.filter_utils import filter_classes


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        # Force shadow/backup to float32 so EMA updates are not lost to fp16 precision
        # under mixed precision training (1 - decay is ~1e-4 which underflows fp16).
        self.shadow_weights = [
            tf.Variable(tf.cast(w, tf.float32), trainable=False, dtype=tf.float32)
            for w in model.trainable_weights
        ]
        self.backup_weights = [
            tf.Variable(tf.cast(w, tf.float32), trainable=False, dtype=tf.float32)
            for w in model.trainable_weights
        ]
        self._applied = False
        
    @tf.function
    def update(self):
        """Update shadow weights with EMA of current training weights."""
        for shadow, weight in zip(self.shadow_weights, self.model.trainable_weights):
            shadow.assign(self.decay * shadow + (1 - self.decay) * tf.cast(weight, shadow.dtype))

    def save(self, path):
        """Save EMA shadow weights to disk so training can resume without losing them."""
        np.savez(path, **{f'w_{i}': w.numpy() for i, w in enumerate(self.shadow_weights)})

    def load(self, path):
        """Restore EMA shadow weights saved by save()."""
        data = np.load(path)
        for i, w in enumerate(self.shadow_weights):
            w.assign(data[f'w_{i}'])
    
    def apply(self):
        """Back up current training weights and load EMA weights into the model."""
        if self._applied:
            return
        for backup, shadow, weight in zip(
            self.backup_weights, self.shadow_weights, self.model.trainable_weights
        ):
            backup.assign(tf.cast(weight, backup.dtype))
            weight.assign(tf.cast(shadow, weight.dtype))
        self._applied = True

    def restore(self):
        """Restore the training weights that were backed up by apply()."""
        if not self._applied:
            return
        for backup, weight in zip(self.backup_weights, self.model.trainable_weights):
            weight.assign(tf.cast(backup, weight.dtype))
        self._applied = False

    def reset_shadow_from_model(self):
        """Re-initialize shadow weights from current model weights (e.g., after resuming)."""
        for shadow, weight in zip(self.shadow_weights, self.model.trainable_weights):
            shadow.assign(tf.cast(weight, shadow.dtype))


class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to 0.

    Implemented as a proper `LearningRateSchedule` so Keras 3 optimizers can
    invoke it with no arguments via `optimizer._learning_rate()` (used by
    weight decay) AND with a step via `__call__(step)` during apply.
    """

    def __init__(self, base_lr, warmup_steps, total_steps, name="WarmupCosineSchedule"):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        # Avoid div-by-zero if warmup_steps == 0 or total == warmup
        warmup_lr = self.base_lr * step / tf.maximum(warmup, 1.0)
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        decay_lr = self.base_lr * 0.5 * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < warmup, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
        }


class WarmupFlatSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by constant learning rate.

    Preferred for short diagnostic runs: the LR never decays to zero, so any
    loss plateau reflects genuine model saturation rather than LR→0.
    """

    def __init__(self, base_lr, warmup_steps, name="WarmupFlatSchedule"):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.base_lr * step / tf.maximum(warmup, 1.0)
        return tf.where(step < warmup, warmup_lr, self.base_lr)

    def get_config(self):
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps, "name": self.name}


def get_learning_rate_schedule(base_lr, warmup_steps, total_steps, schedule_kind='cosine'):
    """Return a Keras LearningRateSchedule.

    schedule_kind: 'cosine' (warmup + cosine decay to 0) or 'flat' (warmup + constant).
    """
    if schedule_kind == 'flat':
        return WarmupFlatSchedule(base_lr, warmup_steps)
    return WarmupCosineSchedule(base_lr, warmup_steps, total_steps)


@tf.function
def train_step(model, x_noisy, timesteps, class_labels, target_F, optimizer):
    """EDM2 training step with preconditioning-cancelled unit-weight loss.

    Preconditioning (c_skip, c_out, c_in) is applied in the dataset so that
    w(σ)·c_out²(σ)=1 exactly — the loss is plain MSE+logvar with no per-sample
    weighting needed here. `target_F` is the preconditioned denoising target
    computed by prepare_batch_conditional_edm.
    """
    with tf.GradientTape() as tape:
        pred_F, logvar = model([x_noisy, timesteps, class_labels], training=True)
        # Reshape (no variables) inherits float16 compute dtype in Keras 3; re-cast here.
        logvar = tf.cast(logvar, tf.float32)
        # Safety clamp: MPLinear-bounded logvar should stay in [-11, 11] by construction,
        # but clamp defensively so a single bad batch cannot cause irreversible divergence.
        logvar = tf.clip_by_value(logvar, -10.0, 5.0)
        target = tf.cast(target_F, pred_F.dtype)
        mse    = tf.cast(tf.square(pred_F - target), tf.float32)
        loss   = tf.reduce_mean(mse / tf.exp(logvar) + logvar)

        # Scale loss INSIDE tape so gradients survive FP16's limited range.
        scaled_loss = optimizer.scale_loss(loss) if hasattr(optimizer, 'scale_loss') else loss

    gradients = tape.gradient(scaled_loss, model.trainable_weights)
    gradients = [
        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else g
        for g in gradients
    ]
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


def _build_probe_model(model):
    """Build a multi-output Keras model for activation magnitude probing.

    Selects all ResNetBlock and attention layer outputs from the functional model.
    Returns (probe_model, probe_names), or (None, []) if no probe layers found.
    """
    probe_layers = [
        (l.name, l.output) for l in model.layers
        if any(tag in l.name for tag in ('res_net_block', 'self_attention', 'sparse_self_attention'))
    ]
    if not probe_layers:
        return None, []
    names = [n for n, _ in probe_layers]
    outputs = [o for _, o in probe_layers]
    probe = keras.Model(inputs=model.inputs, outputs=outputs, name='probe')
    return probe, names


def _collect_diag_step(model, probe_model, probe_names, x_noisy, timesteps, class_labels):
    """Compute per-layer weight and activation magnitude stats for one batch.

    Returns:
        w_mean: dict layer_group -> mean |w| (averaged over all weight tensors in group)
        w_max:  dict layer_group -> max |w|
        a_mean: dict layer_name -> mean |activation|
        a_std:  dict layer_name -> std of activation values
    """
    # Weight stats — group tensors by parent layer (second path component)
    weight_groups = {}
    for w in model.trainable_weights:
        parts = w.path.split('/')
        group = parts[1] if len(parts) > 2 else parts[0]
        abs_w = tf.abs(tf.cast(w, tf.float32))
        g = weight_groups.setdefault(group, {'sum_mean': 0.0, 'max': 0.0, 'n': 0})
        g['sum_mean'] += float(tf.reduce_mean(abs_w))
        g['max'] = max(g['max'], float(tf.reduce_max(abs_w)))
        g['n'] += 1

    w_mean = {k: v['sum_mean'] / v['n'] for k, v in weight_groups.items()}
    w_max  = {k: v['max'] for k, v in weight_groups.items()}

    # Activation stats from probe model
    a_mean, a_std = {}, {}
    if probe_model is not None:
        acts = probe_model([x_noisy, timesteps, class_labels], training=False)
        if not isinstance(acts, (list, tuple)):
            acts = [acts]
        for name, act in zip(probe_names, acts):
            act32 = tf.cast(act, tf.float32)
            a_mean[name] = float(tf.reduce_mean(tf.abs(act32)))
            a_std[name]  = float(tf.math.reduce_std(act32))

    return w_mean, w_max, a_mean, a_std


def _adagn_scale_stats(model):
    """Estimate mean AdaGN scale magnitude across all blocks.

    Feeds a unit-norm synthetic conditioning vector through each scale_shift_mlp
    and computes mean |scale_raw| per block. With the tanh bounding removed,
    scale_raw is the direct output; close to 0 is healthy (near-identity scale).

    Returns:
        mean_scale: float — mean |scale_raw| across all channels and blocks.
            Close to 0 is healthy. Growing past ~2 warrants attention.
        frac_large: float — fraction of blocks where mean |scale_raw| > 2.
    """
    # Weights are at paths like res_net_block_N/ada_gn_M/dense_K/kernel.
    # Match ada_gn Dense layers inside res_net_blocks; exclude the time/class MLP.
    mlp_params = {}  # path_prefix -> {'kernel': Tensor, 'bias': Tensor}
    for w in model.trainable_weights:
        p = w.path
        if 'ada_gn' not in p or 'res_net_block' not in p:
            continue
        prefix, kind = p.rsplit('/', 1)
        mlp_params.setdefault(prefix, {})[kind] = tf.cast(w, tf.float32)

    if not mlp_params:
        return None, None

    scale_means = []
    for params in mlp_params.values():
        K = params.get('kernel')  # [cond_dim, 2*C]
        if K is None:
            continue
        C        = K.shape[-1] // 2
        cond_dim = K.shape[0]
        # Unit-norm synthetic conditioning: approximates a well-normalized
        # conditioning vector as produced by the time/class embedding MLP.
        cond      = tf.ones([1, cond_dim], dtype=tf.float32) / tf.sqrt(float(cond_dim))
        scale_raw = tf.matmul(cond, K[:, :C])
        if (b := params.get('bias')) is not None:
            scale_raw = scale_raw + b[None, :C]
        scale_means.append(float(tf.reduce_mean(tf.abs(scale_raw))))

    mean_scale     = float(np.mean(scale_means))
    frac_large     = float(np.mean([s > 2.0 for s in scale_means]))  # fraction of blocks with mean |scale| > 2
    return mean_scale, frac_large


_FP16_MIN_NORMAL = 6.1e-5  # Smallest positive normal FP16 value


def _gn_precision_risks(a_mean, a_std):
    """Estimate FP16 GroupNorm precision risk per activation layer.

    Zero compute overhead — purely post-processes existing activation stats.

    GroupNorm computes (x - μ) / σ.  In FP16 the quantization unit at magnitude
    M is M / 1024.  When that unit exceeds the within-tensor std, the subtraction
    (x - μ) loses all signal (catastrophic cancellation) and GN outputs noise/zero
    instead of normalised features.

    Risk ratio = (FP16 quant unit at mean magnitude) / observed std
               = a_mean / (1024 × a_std)

    Interpretation:
        < 0.10  — safe, large precision headroom
        0.1–1.0 — marginal, monitor closely
        > 1.0   — cancellation would occur in FP16 (protected by dtype='float32' fix)
    """
    risks = {}
    for name in a_mean:
        std = a_std.get(name, 0.0)
        if std > 1e-8:
            risks[name] = float(a_mean[name]) / (1024.0 * float(std))
    return risks


_GRAD_NORM_MAX_SAMPLES = 4  # Small batch for the diagnostic backward pass


def _collect_grad_norms(model, x_noisy, timesteps, class_labels, true_noise, loss_scale=1.0):
    """Compute per-layer-group gradient norms without applying weight updates.

    Uses only the first _GRAD_NORM_MAX_SAMPLES samples from the batch to bound
    peak GPU memory. The full training batch (e.g. 32) at attention resolution
    32×32 allocates a [B, heads, 1024, 1024] float32 score tensor per attention
    layer (~512 MB each); combined with the GradientTape holding all forward
    activations this OOMs a 16 GB GPU that is already near capacity from training.
    4 samples is sufficient to detect gradient underflow fractions reliably.

    Gradient underflow threshold = FP16_MIN_NORMAL / loss_scale.
    With LossScaleOptimizer scaling gradients up by loss_scale before FP16 storage,
    a true gradient must be below (FP16_MIN_NORMAL / loss_scale) to underflow.
    If loss_scale has been repeatedly halved (overflow events), this threshold rises
    and underflow protection weakens.

    Returns:
        grad_norms:     dict  layer_group -> mean gradient L2 norm
        underflow_frac: dict  layer_group -> fraction of elements below underflow threshold
    """
    n = _GRAD_NORM_MAX_SAMPLES
    x_s  = x_noisy[:n]
    t_s  = timesteps[:n]
    c_s  = class_labels[:n]
    y_s  = true_noise[:n]

    with tf.GradientTape() as tape:
        pred, _ = model([x_s, t_s, c_s], training=False)
        # Cast both sides to float32: pred may be float16 under mixed precision, and a
        # float16 MSE loss readily underflows to 0 → tape.gradient(0, weights) = 0 everywhere,
        # making grad_uflow and grad_p50 meaningless. Float32 loss gives real gradient magnitudes.
        pred_f32 = tf.cast(pred, tf.float32)
        y_f32    = tf.cast(y_s,  tf.float32)
        loss     = tf.reduce_mean(tf.square(pred_f32 - y_f32))
    grads = tape.gradient(loss, model.trainable_weights)

    threshold = _FP16_MIN_NORMAL / max(float(loss_scale), 1.0)

    group_norms = {}
    group_uflow = {}
    for w, g in zip(model.trainable_weights, grads):
        if g is None:
            continue
        parts = w.path.split('/')
        group = parts[1] if len(parts) > 2 else parts[0]
        g32     = tf.cast(g, tf.float32)
        abs_g32 = tf.abs(g32)
        group_norms.setdefault(group, []).append(float(tf.norm(g32)))
        group_uflow.setdefault(group, []).append(
            float(tf.reduce_mean(tf.cast(abs_g32 < threshold, tf.float32)))
        )

    grad_norms     = {k: float(np.mean(v)) for k, v in group_norms.items()}
    underflow_frac = {k: float(np.mean(v)) for k, v in group_uflow.items()}
    return grad_norms, underflow_frac


def _save_diag_history(history, path):
    """Persist diagnostic history as a flat npz. Keys use '__' as namespace separator."""
    flat = {'steps': np.array(history['steps'], dtype=np.int32)}
    for layer, vals in history['weight_mean'].items():
        flat[f'wmean__{layer}'] = np.array(vals, dtype=np.float32)
    for layer, vals in history['weight_max'].items():
        flat[f'wmax__{layer}'] = np.array(vals, dtype=np.float32)
    for layer, vals in history['act_mean'].items():
        flat[f'amean__{layer}'] = np.array(vals, dtype=np.float32)
    for layer, vals in history['act_std'].items():
        flat[f'astd__{layer}'] = np.array(vals, dtype=np.float32)
    for layer, vals in history.get('gn_risk', {}).items():
        flat[f'gnrisk__{layer}'] = np.array(vals, dtype=np.float32)
    if history.get('loss_scale'):
        flat['scalar__loss_scale'] = np.array(history['loss_scale'], dtype=np.float32)
    for layer, vals in history.get('grad_norm', {}).items():
        flat[f'gnorm__{layer}'] = np.array(vals, dtype=np.float32)
    for layer, vals in history.get('grad_underflow', {}).items():
        flat[f'guflow__{layer}'] = np.array(vals, dtype=np.float32)
    for layer, vals in history.get('weight_delta', {}).items():
        flat[f'wdelta__{layer}'] = np.array(vals, dtype=np.float32)
    np.savez(path, **flat)


def generate_samples(model, config, num_samples=16, guidance_scale=3.0):
    """Generate monitoring samples via the EDM Heun ODE sampler from pure noise.

    No seed image required — starts from x ~ N(0, sigma_max²·I).
    Uses 20 denoising steps (sufficient for monitoring; increase at inference time).
    """
    from diffusion.diffusion_edm_sample import sample_edm_batch
    sigmas = diffusion_utils.edm_sigma_schedule(
        sigma_max=config['sigma_max'],
        sigma_min=config['sigma_min'],
        num_steps=20,
        rho=config.get('sigma_rho', 7),
    )
    class_labels = (np.arange(num_samples) % config['num_classes']).astype(np.int32)
    samples = sample_edm_batch(
        model,
        class_labels=class_labels,
        num_classes=config['num_classes'],
        sigmas=sigmas,
        sigma_data=config['sigma_data'],
        guidance_scale=guidance_scale,
        image_size=config['image_size'],
    )
    return samples[..., 0]  # [N, H, W]


def save_sample_grid(samples, step, output_dir, run_id='', seed_images=None):
    """Save grid of generated samples, optionally with seed images interleaved for comparison.

    When seed_images is provided, rows alternate: seeds then generated for each group of 4,
    making it easy to compare a seed image with its denoised output directly below it.
    """
    n_samples = len(samples)
    n_cols = 4
    sample_rows = (n_samples + n_cols - 1) // n_cols

    if seed_images is not None:
        # Interleaved: for each group of n_cols, seeds row then generated row
        fig, axes = plt.subplots(sample_rows * 2, n_cols, figsize=(12, 6 * sample_rows))
        axes = axes.reshape(sample_rows * 2, n_cols)

        for idx in range(n_samples):
            seed_row = (idx // n_cols) * 2
            col = idx % n_cols
            axes[seed_row, col].imshow(seed_images[idx], cmap='viridis')
            axes[seed_row, col].set_title(f'Seed {idx}', fontsize=9)
            axes[seed_row, col].axis('off')
            axes[seed_row + 1, col].imshow(samples[idx], cmap='viridis')
            axes[seed_row + 1, col].set_title(f'Gen {idx}', fontsize=9)
            axes[seed_row + 1, col].axis('off')

        for idx in range(n_samples, sample_rows * n_cols):
            seed_row = (idx // n_cols) * 2
            axes[seed_row, idx % n_cols].axis('off')
            axes[seed_row + 1, idx % n_cols].axis('off')
    else:
        fig, axes = plt.subplots(sample_rows, n_cols, figsize=(12, 3 * sample_rows))
        axes = axes.reshape(sample_rows, n_cols)

        for idx, sample in enumerate(samples):
            axes[idx // n_cols, idx % n_cols].imshow(sample, cmap='viridis')
            axes[idx // n_cols, idx % n_cols].set_title(f'Class {idx}', fontsize=10)
            axes[idx // n_cols, idx % n_cols].axis('off')

        for idx in range(n_samples, sample_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')

    plt.suptitle(f'Generated Samples at Step {step}', fontsize=14)
    plt.tight_layout()

    suffix = f'_{run_id}' if run_id else ''
    output_path = os.path.join(output_dir, f'samples_step_{step:06d}{suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved samples to {output_path}")


def plot_training_history(losses, lrs, output_path):
    """Plot training loss and learning rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(losses)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Learning rate
    ax2.plot(lrs)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def train(config, resume_from=None):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        resume_from: Path to checkpoint to resume from (optional)
    """
    print_config(config)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"  Run ID: {run_id}")

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    # Enable mixed precision if configured
    if config.get('mixed_precision', False):
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision training enabled (FP16)")
    
    # Load data
    print("\n📊 Loading data...")
    data_dir = config['data_dir']
    # Cast features to float32 up-front: halves dataset memory vs. native float64
    # and matches the model's compute dtype.
    X_train = np.load(os.path.join(data_dir, config['feature_file'])).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, config['label_file'])).astype(np.float32)
    
    print(f"  Features: {X_train.shape}")
    print(f"  Labels: {y_train.shape}")

    excluded = config.get('excluded_classes', [])
    if excluded:
        X_train, y_train = filter_classes(X_train, y_train, excluded)
        print(f"  Excluded classes {excluded}: {len(X_train)} samples remain")

    # Set global normalization constants (and persist them for sampling)
    diffusion_utils.set_data_range(X_train.min(), X_train.max())
    print(f"  Data range: [{diffusion_utils.DATA_MIN:.6f}, {diffusion_utils.DATA_MAX:.6f}]")

    norm_path = os.path.join(config['checkpoint_dir'], 'norm_constants.json')
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    diffusion_utils.save_norm_constants(norm_path)

    # Create EDM2 dataset — σ sampled from log-normal each batch, preconditioning applied inline
    print("\n🔄 Creating dataset...")
    dataset = diffusion_utils.prepare_dataset_conditional_edm(
        X_train,
        y_train,
        num_classes=config['num_classes'],
        batch_size=config['batch_size'],
        dropout_rate=config['dropout_rate'],
        shuffle=True,
        drop_remainder=True,
        excluded_classes=config.get('excluded_classes', None),
        P_mean=config.get('P_mean', -2.0),
        P_std=config.get('P_std', 1.2),
        sigma_min=config.get('sigma_min', 0.002),
        sigma_max=config.get('sigma_max', 80.0),
        sigma_data=float(config.get('sigma_data', 0.139)),
    )
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Classifier-free dropout: {config['dropout_rate']*100:.0f}%")
    print(f"  EDM σ sampling: P_mean={config.get('P_mean')}  P_std={config.get('P_std')}  σ∈[{config.get('sigma_min')}, {config.get('sigma_max')}]")
    if config.get('excluded_classes'):
        print(f"  Excluded classes: {config['excluded_classes']}")

    # Build model
    print("\n🏗️  Building model...")
    model = build_unet(config)
    print(f"  Parameters: {model.count_params():,}")
    
    # Create optimizer with learning rate schedule
    lr_schedule = get_learning_rate_schedule(
        config['learning_rate'],
        config['warmup_steps'],
        config['num_steps'],
        schedule_kind=config.get('lr_schedule', 'cosine'),
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.99,
        clipnorm=config['gradient_clip']
    )
    mixed_precision_enabled = config.get('mixed_precision', False)
    if mixed_precision_enabled:
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Create EMA
    ema = EMA(model, decay=config['ema_decay'])

    # Diagnostic setup — probe model for activation magnitudes
    diag_interval = config.get('diag_interval', 999_999)
    probe_model, probe_names = _build_probe_model(model)
    ckpt_dir = config['checkpoint_dir']
    if 'diagnostic' in ckpt_dir:
        diag_prefix = 'diffusion_diagnose_'
    elif 'local' in ckpt_dir:
        diag_prefix = 'diffusion_local_'
    else:
        diag_prefix = 'diffusion_'
    diag_path = os.path.join(ckpt_dir, f'{diag_prefix}{run_id}_magnitudes.npz')
    diag_history = {
        'steps': [], 'weight_mean': {}, 'weight_max': {}, 'act_mean': {}, 'act_std': {},
        'gn_risk': {}, 'loss_scale': [], 'grad_norm': {}, 'grad_underflow': {},
        'weight_delta': {},
    }
    if diag_interval < 999_999:
        print(f"  Diagnostic magnitude logging every {diag_interval} steps → {diag_path}")

    # Resume from checkpoint if specified
    start_step = 0
    if resume_from:
        print(f"\n📂 Resuming from checkpoint: {resume_from}")
        model.load_weights(resume_from)

        # Try to also restore EMA shadow (saved alongside checkpoint)
        ema_path = resume_from.replace('.weights.h5', '.ema.npz')
        if os.path.exists(ema_path):
            ema.load(ema_path)
            print(f"  Restored EMA shadow from: {ema_path}")
        else:
            ema.reset_shadow_from_model()
            print("  ⚠️  No EMA shadow file found; shadow initialized from model weights.")

        # Best-effort: extract numeric step from filename, else start at 0
        try:
            tail = resume_from.rsplit('_', 1)[-1].split('.')[0]
            start_step = int(tail)
        except (ValueError, IndexError):
            start_step = 0
        print(f"  Resuming from step {start_step}")
    
    # Training loop
    print("\n🚀 Starting training...")
    print(f"  Total steps: {config['num_steps']:,}")
    print(f"  Save interval: {config['save_interval']:,}")
    print(f"  Sample interval: {config['sample_interval']:,}")
    print(f"  Log interval: {config['log_interval']:,}")
    print(f"  sigma_data: {config.get('sigma_data', 0.139)}")

    losses = []
    lrs = []
    step = start_step
    prev_checkpoint_path = None      # track for deletion after next save
    prev_diag_checkpoint_path = None # track diag rolling checkpoint for deletion
    w_snapshot = {}                  # weight snapshots for per-diag-step delta tracking

    dataset_iter = iter(dataset.repeat())

    while step < config['num_steps']:
        # Get batch
        inputs, true_noise = next(dataset_iter)

        # Training step
        loss = train_step(
            model,
            inputs['X_noisy'],
            inputs['timesteps'],
            inputs['class_labels'],
            true_noise,
            optimizer,
        )
        
        # Update EMA
        ema.update()

        # Magnitude diagnostics
        if step % diag_interval == 0:
            w_mean, w_max, a_mean, a_std = _collect_diag_step(
                model, probe_model, probe_names,
                inputs['X_noisy'], inputs['timesteps'], inputs['class_labels'],
            )
            diag_history['steps'].append(step)
            for k, v in w_mean.items():
                diag_history['weight_mean'].setdefault(k, []).append(v)
            for k, v in w_max.items():
                diag_history['weight_max'].setdefault(k, []).append(v)
            for k, v in a_mean.items():
                diag_history['act_mean'].setdefault(k, []).append(v)
            for k, v in a_std.items():
                diag_history['act_std'].setdefault(k, []).append(v)

            # FP16 precision diagnostics — zero-overhead GN risk + one extra backward pass
            # TF/Keras versions differ on the attribute name for the current loss scale.
            # Try public then private names; stop at the first one that yields a float.
            # Final fallback: scale_loss(1.0) returns the scale directly and works across
            # all Keras versions (the attribute scan fails silently on Keras 3).
            raw_ls = None
            for _attr in ('loss_scale_factor', 'loss_scale', '_loss_scale', '_current_scale'):
                _val = getattr(optimizer, _attr, None)
                if _val is not None:
                    try:
                        raw_ls = float(_val)
                    except Exception:
                        pass
                    else:
                        break
            if raw_ls is None and hasattr(optimizer, 'scale_loss'):
                try:
                    raw_ls = float(optimizer.scale_loss(tf.constant(1.0, dtype=tf.float32)))
                except Exception:
                    pass
            loss_scale = raw_ls if raw_ls is not None else 1.0
            gn_risks = _gn_precision_risks(a_mean, a_std)
            grad_norms, uflow_frac = _collect_grad_norms(
                model, inputs['X_noisy'], inputs['timesteps'], inputs['class_labels'],
                true_noise, loss_scale=loss_scale,
            )
            diag_history['loss_scale'].append(loss_scale)
            for k, v in gn_risks.items():
                diag_history['gn_risk'].setdefault(k, []).append(v)
            for k, v in grad_norms.items():
                diag_history['grad_norm'].setdefault(k, []).append(v)
            for k, v in uflow_frac.items():
                diag_history['grad_underflow'].setdefault(k, []).append(v)

            # Weight delta: mean |w_now - w_prev| per group since last diag step
            w_delta_by_group = {}
            for w in model.trainable_weights:
                parts = w.path.split('/')
                group = parts[1] if len(parts) > 2 else parts[0]
                w32 = tf.cast(w, tf.float32).numpy()
                prev = w_snapshot.get(w.path)
                if prev is not None:
                    w_delta_by_group.setdefault(group, []).append(float(np.mean(np.abs(w32 - prev))))
                w_snapshot[w.path] = w32
            w_delta_mean = {k: float(np.mean(v)) for k, v in w_delta_by_group.items()}
            for k, v in w_delta_mean.items():
                diag_history['weight_delta'].setdefault(k, []).append(v)

            _save_diag_history(diag_history, diag_path)

            # Save rolling checkpoint so a killed run still has recent weights
            diag_ckpt = os.path.join(ckpt_dir, f'{diag_prefix}{run_id}_step_{step:06d}.weights.h5')
            model.save_weights(diag_ckpt)
            ema.save(diag_ckpt.replace('.weights.h5', '.ema.npz'))
            if prev_diag_checkpoint_path and os.path.exists(prev_diag_checkpoint_path):
                os.remove(prev_diag_checkpoint_path)
                ema_prev = prev_diag_checkpoint_path.replace('.weights.h5', '.ema.npz')
                if os.path.exists(ema_prev):
                    os.remove(ema_prev)
            prev_diag_checkpoint_path = diag_ckpt

            all_wm = list(w_mean.values())
            all_am = list(a_mean.values()) if a_mean else [float('nan')]
            mean_scale, frac_large = _adagn_scale_stats(model)
            scale_str = f"  adagn_|scale|={mean_scale:.3f}  large_frac={frac_large:.2f}" if mean_scale is not None else ""
            max_gn_risk  = max(gn_risks.values())  if gn_risks   else float('nan')
            mean_uflow   = np.mean(list(uflow_frac.values())) if uflow_frac else float('nan')
            wdelta_str   = f"  mean_Δ|w|={np.mean(list(w_delta_mean.values())):.2e}" if w_delta_mean else "  mean_Δ|w|=n/a"
            print(
                f"  [diag] mean_|w|={np.mean(all_wm):.4f}  max_|w|={max(w_max.values()):.4f}"
                f"  mean_|act|={np.mean(all_am):.4f}{scale_str}"
                f"  gn_risk={max_gn_risk:.3f}  loss_scale={loss_scale:.0f}(raw:{raw_ls})  grad_uflow={mean_uflow:.3f}"
                f"{wdelta_str}  ckpt→{os.path.basename(diag_ckpt)}"
            )

        # Track metrics
        losses.append(float(loss))
        lrs.append(float(lr_schedule(step)))

        step += 1

        # Logging
        if step % config['log_interval'] == 0:
            print(f"Step {step:6d}/{config['num_steps']:6d} | Loss: {loss:.6f} | LR: {lrs[-1]:.6f}")
        
        # Save checkpoint
        if step % config['save_interval'] == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'diffusion_model_step_{step:06d}.weights.h5')
            model.save_weights(checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")

            # Delete previous intermediate checkpoint to keep disk usage flat
            if prev_checkpoint_path and os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)
                print(f"  🗑️  Removed old checkpoint: {prev_checkpoint_path}")
            prev_checkpoint_path = checkpoint_path

            # Save training history (suffixed by run_id — one file per run, overwritten each interval)
            history_path = os.path.join(config['checkpoint_dir'], f'training_history_{run_id}.npz')
            np.savez(history_path, losses=losses, lrs=lrs)

            plot_path = os.path.join(config['checkpoint_dir'], f'training_history_{run_id}.png')
            plot_training_history(losses, lrs, plot_path)
        
        # Generate samples
        if step % config['sample_interval'] == 0:
            print(f"  🎨 Generating samples...")
            ema.apply()  # Use EMA weights for generation
            samples = generate_samples(model, config, num_samples=16, guidance_scale=3.0)
            save_sample_grid(samples, step, config['sample_dir'], run_id=run_id)
            ema.restore()  # Restore training weights
    
    # Final save
    print("\n✅ Training complete!")
    final_checkpoint = os.path.join(config['checkpoint_dir'], 'diffusion_model_final.weights.h5')
    model.save_weights(final_checkpoint)
    print(f"  💾 Saved final checkpoint: {final_checkpoint}")
    
    # Save EMA weights (apply shadow → model, save, restore training weights)
    ema.apply()
    ema_checkpoint = os.path.join(config['checkpoint_dir'], 'diffusion_model_ema.weights.h5')
    model.save_weights(ema_checkpoint)
    ema.restore()
    print(f"  💾 Saved EMA checkpoint: {ema_checkpoint}")
    
    # Final training history
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.npz')
    np.savez(history_path, losses=losses, lrs=lrs)
    plot_path = os.path.join(config['checkpoint_dir'], 'training_history.png')
    plot_training_history(losses, lrs, plot_path)
    print(f"  📊 Saved training history: {history_path}")

    # Final magnitude diagnostic flush
    if diag_history['steps']:
        _save_diag_history(diag_history, diag_path)
        print(f"  📊 Saved magnitude history: {diag_path}")
    
    print(f"\n📈 Final loss: {losses[-1]:.6f}")
    print(f"📈 Average loss (last 1000 steps): {np.mean(losses[-1000:]):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Conditional DDPM with Classifier-Free Guidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train locally (Mac M2)
  python diffusion_train.py --mode local
  
  # Train on remote (A100)
  python diffusion_train.py --mode remote
  
  # Resume from checkpoint
  python diffusion_train.py --mode local --resume output/diffusion/local/checkpoints/diffusion_model_step_010000.weights.h5
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'remote', 'diagnostic'],
        default='local',
        help='Training mode: local (Mac M2), remote (A100), or diagnostic (1000-step remote architecture)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.mode)
    
    # Train
    train(config, resume_from=args.resume)
