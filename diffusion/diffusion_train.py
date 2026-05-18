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


def get_learning_rate_schedule(base_lr, warmup_steps, total_steps):
    """Return a Keras `LearningRateSchedule` (warmup + cosine decay)."""
    return WarmupCosineSchedule(base_lr, warmup_steps, total_steps)


@tf.function
def train_step(model, x_noisy, timesteps, class_labels, true_noise, optimizer):
    """Single training step — uniform MSE loss across all timesteps.

    Keras 3 LossScaleOptimizer handles FP16 gradient scaling internally in
    apply_gradients, so no mixed_precision flag is needed here.
    """
    with tf.GradientTape() as tape:
        predicted_noise = model([x_noisy, timesteps, class_labels], training=True)
        loss = tf.reduce_mean(tf.square(predicted_noise - tf.cast(true_noise, predicted_noise.dtype)))
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


def generate_samples(model, config, num_samples=16, guidance_scale=3.0, eps_threshold=0.0):
    """
    Generate sample images during training for monitoring.

    eps_threshold is intentionally not read from config here — checkpoint samples
    should always use 0.0 so the images reflect actual model state rather than
    being affected by the threshold suppressing early-stage low-magnitude predictions.
    """
    from diffusion.diffusion_sample import sample_ddpm_batch

    num_classes = config['num_classes']
    class_labels = (np.arange(num_samples) % num_classes).astype(np.int32)

    samples = sample_ddpm_batch(
        model,
        class_labels=class_labels,
        num_classes=num_classes,
        guidance_scale=guidance_scale,
        num_steps=config['timesteps'],
        image_size=config['image_size'],
        eps_threshold=eps_threshold,
    )
    return samples[..., 0]  # drop channel dim: [N, H, W]


def save_sample_grid(samples, step, output_dir, run_id=''):
    """Save grid of generated samples."""
    n_samples = len(samples)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, sample in enumerate(samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.imshow(sample, cmap='viridis')
        ax.set_title(f'Class {idx}', fontsize=10)
        ax.axis('off')

    for idx in range(n_samples, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

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

    # Optional log1p preprocessing (compresses sparse / heavy-tailed distributions)
    if config.get('log_transform', False):
        diffusion_utils.configure_log_transform(X_train, enable=True)
        print(f"  Log range:  [{diffusion_utils.LOG_MIN:.6f}, {diffusion_utils.LOG_MAX:.6f}] (log1p enabled)")

    norm_path = os.path.join(config['checkpoint_dir'], 'norm_constants.json')
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    diffusion_utils.save_norm_constants(norm_path)

    # Initialize variance schedule (cosine or linear)
    schedule_kind = config.get('variance_schedule', 'cosine')
    alpha, alpha_cumprod, beta = diffusion_utils.init_schedule(config['timesteps'], kind=schedule_kind)
    print(f"  Variance schedule: kind={schedule_kind}, T={config['timesteps']}, alpha_bar range=[{alpha_cumprod.min():.6f}, {alpha_cumprod.max():.6f}]")

    # Create dataset
    print("\n🔄 Creating dataset...")
    dataset = diffusion_utils.prepare_dataset_conditional(
        X_train,
        y_train,
        num_classes=config['num_classes'],
        batch_size=config['batch_size'],
        dropout_rate=config['dropout_rate'],
        shuffle=True,
        drop_remainder=True,
        excluded_classes=config.get('excluded_classes', None),
    )
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Classifier-free dropout: {config['dropout_rate']*100:.0f}%")
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
        config['num_steps']
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=config['gradient_clip']
    )
    mixed_precision_enabled = config.get('mixed_precision', False)
    if mixed_precision_enabled:
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Create EMA
    ema = EMA(model, decay=config['ema_decay'])
    
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
    
    losses = []
    lrs = []
    step = start_step
    prev_checkpoint_path = None  # track for deletion after next save

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
