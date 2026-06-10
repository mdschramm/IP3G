"""
Sampling and generation script for Conditional DDPM.

Implements:
- DDPM sampling with classifier-free guidance
- Batch generation for data augmentation
- Guidance scale sweep
- Sample visualization
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusion.diffusion_config import get_config
from diffusion.diffusion_model import build_unet
import diffusion.diffusion_utils as diffusion_utils


@tf.function(reduce_retracing=True)
def _denoise_step(
    model,
    x,
    t_tensor,
    cond_labels,
    uncond_labels,
    guidance_scale,
    alpha_t,
    alpha_bar_t,
    sigma_t,
    add_noise,
    eps_threshold,
):
    """Single reverse-diffusion step for a batch under classifier-free guidance.

    Runs cond + uncond as a single concatenated forward pass for efficiency.

    Args:
        eps_threshold: Soft-threshold magnitude on predicted noise. 0.0 disables.
            Pushes |eps| < threshold to 0, encouraging sparse predictions.
    """
    batch = tf.shape(x)[0]
    x_in = tf.concat([x, x], axis=0)
    t_in = tf.concat([t_tensor, t_tensor], axis=0)
    labels_in = tf.concat([uncond_labels, cond_labels], axis=0)

    eps_both, _ = model([x_in, t_in, labels_in], training=False)
    eps_uncond = eps_both[:batch]
    eps_cond = eps_both[batch:]
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # Soft-threshold the predicted noise to encourage sparsity in the recovered
    # signal. Disabled when eps_threshold == 0 (returns eps unchanged).
    eps = tf.sign(eps) * tf.nn.relu(tf.abs(eps) - eps_threshold)

    mean = (1.0 / tf.sqrt(alpha_t)) * (
        x - ((1.0 - alpha_t) / tf.sqrt(1.0 - alpha_bar_t)) * eps
    )

    noise = tf.random.normal(tf.shape(x))
    return tf.where(add_noise, mean + sigma_t * noise, mean)


def sample_ddpm_batch(
    model,
    class_labels,
    num_classes,
    guidance_scale=3.0,
    num_steps=1000,
    image_size=128,
    denormalize_output=True,
    eps_threshold=0.0,
):
    """Generate a batch of images with DDPM + classifier-free guidance.

    Args:
        model: Trained U-Net.
        class_labels: Iterable/array of class indices, length N.
        num_classes: Total number of classes. The unconditional token is ``num_classes``.
        guidance_scale: CFG scale w.
        num_steps: Number of diffusion timesteps (must match training schedule).
        image_size: Spatial size of the square image.
        denormalize_output: When True, map samples back to the original data range
            using ``diffusion_utils.DATA_MIN/DATA_MAX``. Otherwise return values in [0, 1].
        eps_threshold: Soft-threshold magnitude on predicted noise (0.0 = disabled).
            For sparse data, small values like 0.05 push near-zero predictions to
            exact zero, reducing the residual fog seen in dense generations.

    Returns:
        numpy array shape [N, H, W, C].
    """
    class_labels = np.asarray(class_labels, dtype=np.int32)
    batch = class_labels.shape[0]

    # Start from pure noise
    x = tf.random.normal([batch, image_size, image_size, 1])

    # Variance schedule as tensors (set once by diffusion_utils.init_schedule)
    alpha = tf.constant(diffusion_utils.alpha, dtype=tf.float32)
    alpha_cumprod = tf.constant(diffusion_utils.alpha_cumprod, dtype=tf.float32)
    beta = tf.constant(diffusion_utils.beta, dtype=tf.float32)

    cond_labels = tf.constant(class_labels, dtype=tf.int32)
    uncond_labels = tf.fill([batch], tf.constant(num_classes, dtype=tf.int32))
    guidance_scale_t = tf.constant(guidance_scale, dtype=tf.float32)
    eps_threshold_t = tf.constant(eps_threshold, dtype=tf.float32)

    for t in tqdm(range(num_steps, 0, -1), desc='Sampling', leave=False):
        t_tensor = tf.fill([batch], tf.constant(t, dtype=tf.int32))
        x = _denoise_step(
            model,
            x,
            t_tensor,
            cond_labels,
            uncond_labels,
            guidance_scale_t,
            alpha[t],
            alpha_cumprod[t],
            tf.sqrt(beta[t]),
            tf.constant(t > 1),
            eps_threshold_t,
        )

    x = tf.clip_by_value(x, 0.0, 1.0)  # clipped ReLU: zeros negatives, caps at 1
    if denormalize_output:
        x = diffusion_utils.denormalize(x)
    # else: already [0, 1]
    return x.numpy()


def sample_ddpm(model, class_label, num_classes, guidance_scale=3.0, num_steps=1000, image_size=128, eps_threshold=0.0):
    """Generate a single image for one class.

    ``num_classes`` is required so the unconditional CFG token (``num_classes``)
    matches the model's training-time token and doesn't silently alias a real class.
    """
    out = sample_ddpm_batch(
        model,
        class_labels=[class_label],
        num_classes=num_classes,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        image_size=image_size,
        eps_threshold=eps_threshold,
    )
    return tf.convert_to_tensor(out)  # shape [1, H, W, 1]


def generate_batch(model, class_label, num_samples, num_classes, guidance_scale=3.0, num_steps=1000, image_size=128, eps_threshold=0.0):
    """Generate ``num_samples`` images of a single class as one batched call."""
    class_labels = np.full((num_samples,), int(class_label), dtype=np.int32)
    samples = sample_ddpm_batch(
        model,
        class_labels=class_labels,
        num_classes=num_classes,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        image_size=image_size,
        eps_threshold=eps_threshold,
    )
    return samples[..., 0]  # [N, H, W]


def generate_dataset(model, config, samples_per_class=100, guidance_scale=3.0, output_dir=None):
    """
    Generate synthetic dataset for all classes.
    
    Args:
        model: Trained model
        config: Configuration dictionary
        samples_per_class: Number of samples per class
        guidance_scale: Guidance scale
        output_dir: Directory to save generated data
        
    Returns:
        X_synthetic: Generated images, shape [num_classes * samples_per_class, H, W]
        y_synthetic: One-hot labels, shape [num_classes * samples_per_class, num_classes]
    """
    num_classes = config['num_classes']
    image_size = config['image_size']
    num_steps = config['timesteps']
    eps_threshold = config.get('eps_threshold', 0.0)
    
    print(f"\n🎨 Generating synthetic dataset:")
    print(f"  Classes: {num_classes}")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Total samples: {num_classes * samples_per_class}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Epsilon threshold: {eps_threshold}")
    
    all_images = []
    all_labels = []

    for class_id in tqdm(range(num_classes), desc='Classes'):
        # Generate all samples for this class in a single batched pass
        samples = generate_batch(
            model,
            class_id,
            samples_per_class,
            num_classes=num_classes,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            image_size=image_size,
            eps_threshold=eps_threshold,
        )

        # Create one-hot labels (match float32 dtype of real labels)
        labels = np.zeros((samples_per_class, num_classes), dtype=np.float32)
        labels[:, class_id] = 1.0

        all_images.append(samples)
        all_labels.append(labels)
    
    # Concatenate all classes
    X_synthetic = np.concatenate(all_images, axis=0)
    y_synthetic = np.concatenate(all_labels, axis=0)
    
    # Shuffle
    indices = np.arange(len(X_synthetic))
    np.random.shuffle(indices)
    X_synthetic = X_synthetic[indices]
    y_synthetic = y_synthetic[indices]
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        features_path = os.path.join(output_dir, f'diffusion_synthetic_expressions_w{guidance_scale:.1f}.npy')
        labels_path = os.path.join(output_dir, f'diffusion_synthetic_labels_w{guidance_scale:.1f}.npy')
        
        np.save(features_path, X_synthetic)
        np.save(labels_path, y_synthetic)
        
        print(f"\n💾 Saved synthetic dataset:")
        print(f"  Features: {features_path}")
        print(f"  Labels: {labels_path}")
        print(f"  Shape: {X_synthetic.shape}")
        print(f"  Value range: [{X_synthetic.min():.4f}, {X_synthetic.max():.4f}]")
    
    return X_synthetic, y_synthetic


def guidance_scale_sweep(model, config, class_label, guidance_scales=[1.0, 2.0, 3.0, 5.0, 7.5], output_dir=None):
    """
    Generate samples with different guidance scales to visualize effect.
    
    Args:
        model: Trained model
        config: Configuration dictionary
        class_label: Class to generate
        guidance_scales: List of guidance scales to test
        output_dir: Directory to save visualization
    """
    print(f"\n🔍 Guidance scale sweep for class {class_label}:")
    
    samples = []
    for w in guidance_scales:
        print(f"  Generating with w={w}...")
        sample = sample_ddpm(
            model,
            class_label,
            guidance_scale=w,
            num_steps=config['timesteps'],
            image_size=config['image_size'],
            num_classes=config['num_classes'],
        )
        samples.append(sample[0, :, :, 0].numpy())
    
    # Visualize
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(4 * len(guidance_scales), 4))
    
    # Normalize axes to 1D array even when len(guidance_scales) == 1
    axes = np.atleast_1d(axes)
    for idx, (w, sample) in enumerate(zip(guidance_scales, samples)):
        axes[idx].imshow(sample, cmap='viridis')
        axes[idx].set_title(f'w = {w}', fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle(f'Guidance Scale Sweep - Class {class_label}', fontsize=14)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'guidance_sweep_class_{class_label}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved to {output_path}")
    
    plt.show()


def visualize_class_samples(model, config, num_classes_to_show=16, samples_per_class=4, guidance_scale=3.0, output_dir=None):
    """
    Generate and visualize samples from multiple classes.
    
    Args:
        model: Trained model
        config: Configuration dictionary
        num_classes_to_show: Number of classes to visualize
        samples_per_class: Samples per class
        guidance_scale: Guidance scale
        output_dir: Directory to save visualization
    """
    print(f"\n🎨 Generating samples from {num_classes_to_show} classes...")
    
    fig, axes = plt.subplots(num_classes_to_show, samples_per_class,
                             figsize=(3 * samples_per_class, 3 * num_classes_to_show),
                             squeeze=False)

    num_classes = config['num_classes']
    for class_id in range(num_classes_to_show):
        # Generate samples_per_class images for this class in one batched call
        batch = sample_ddpm_batch(
            model,
            class_labels=np.full((samples_per_class,), class_id, dtype=np.int32),
            num_classes=num_classes,
            guidance_scale=guidance_scale,
            num_steps=config['timesteps'],
            image_size=config['image_size'],
        )
        for sample_idx in range(samples_per_class):
            ax = axes[class_id, sample_idx]
            ax.imshow(batch[sample_idx, :, :, 0], cmap='viridis')
            if sample_idx == 0:
                ax.set_ylabel(f'Class {class_id}', fontsize=10)
            ax.axis('off')
    
    plt.suptitle(f'Generated Samples (w={guidance_scale})', fontsize=14)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'class_samples_w{guidance_scale:.1f}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved to {output_path}")
    
    plt.show()


def load_model(config, checkpoint_path):
    """Load trained model from checkpoint and initialize sampler state."""
    print(f"\n📂 Loading model from: {checkpoint_path}")

    # Initialize variance schedule (must match training schedule kind)
    schedule_kind = config.get('variance_schedule', 'cosine')
    diffusion_utils.init_schedule(config['timesteps'], kind=schedule_kind)

    # Try to restore data normalization constants saved during training so that
    # generated samples live on the same scale as the original dataset.
    norm_path = os.path.join(config['checkpoint_dir'], 'norm_constants.json')
    if os.path.exists(norm_path):
        diffusion_utils.load_norm_constants(norm_path)
        print(f"  Loaded normalization constants: min={diffusion_utils.DATA_MIN:.6f}, max={diffusion_utils.DATA_MAX:.6f}")
    else:
        print("  ⚠️  norm_constants.json not found; samples will be returned in [0, 1].")

    # Build model and load weights
    model = build_unet(config)
    model.load_weights(checkpoint_path)

    print(f"✅ Model loaded successfully")
    print(f"  Parameters: {model.count_params():,}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate samples from trained Conditional DDPM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full dataset (100 samples per class)
  python diffusion_sample.py --mode local --checkpoint output/diffusion/local/checkpoints/diffusion_model_ema.weights.h5 --generate-dataset --samples-per-class 100
  
  # Visualize samples from different classes
  python diffusion_sample.py --mode local --checkpoint output/diffusion/local/checkpoints/diffusion_model_ema.weights.h5 --visualize
  
  # Guidance scale sweep
  python diffusion_sample.py --mode local --checkpoint output/diffusion/local/checkpoints/diffusion_model_ema.weights.h5 --guidance-sweep --class-id 0
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'remote'],
        default='local',
        help='Configuration mode'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--generate-dataset',
        action='store_true',
        help='Generate full synthetic dataset'
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=100,
        help='Number of samples per class for dataset generation'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=3.0,
        help='Guidance scale (w)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize samples from multiple classes'
    )
    parser.add_argument(
        '--guidance-sweep',
        action='store_true',
        help='Perform guidance scale sweep'
    )
    parser.add_argument(
        '--class-id',
        type=int,
        default=0,
        help='Class ID for guidance sweep or single sample generation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for generated data (default: derived from --mode config sample_dir)'
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.mode)
    
    # Load model
    model = load_model(config, args.checkpoint)
    
    # Execute requested action
    if args.generate_dataset:
        generate_dataset(
            model,
            config,
            samples_per_class=args.samples_per_class,
            guidance_scale=args.guidance_scale,
            output_dir=args.output_dir
        )
    
    elif args.visualize:
        visualize_class_samples(
            model,
            config,
            num_classes_to_show=16,
            samples_per_class=4,
            guidance_scale=args.guidance_scale,
            output_dir=args.output_dir
        )
    
    elif args.guidance_sweep:
        guidance_scale_sweep(
            model,
            config,
            class_label=args.class_id,
            guidance_scales=[1.0, 2.0, 3.0, 5.0, 7.5],
            output_dir=args.output_dir
        )
    
    else:
        print("\n⚠️  No action specified. Use --generate-dataset, --visualize, or --guidance-sweep")
        print("   Run with --help for usage examples")
