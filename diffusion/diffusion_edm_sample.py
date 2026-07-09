"""
EDM 2nd-order Heun ODE sampler for conditional generation from pure Gaussian noise.

Implements Algorithm 1 of Karras et al. 2022 ("Elucidating the Design Space of
Diffusion-Based Generative Models") with classifier-free guidance.

Two model evaluations per step (Euler predictor + Heun corrector), except at
the final step (σ_next = 0) where only the Euler step fires to avoid a divide-
by-zero in the ODE direction.

Usage:
    from diffusion.diffusion_edm_sample import sample_edm_batch
    from diffusion.diffusion_utils import edm_sigma_schedule

    sigmas  = edm_sigma_schedule(sigma_max=80.0, sigma_min=0.002, num_steps=40)
    samples = sample_edm_batch(
        model,
        class_labels=np.array([0, 1, 5, 12]),
        num_classes=54,
        sigmas=sigmas,
        sigma_data=0.139,
        guidance_scale=3.0,
        image_size=128,
    )  # → numpy [N, H, W, 1] in [0, 1]
"""

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import diffusion.diffusion_utils as diffusion_utils


def _denoise(model, x, sigma, cond_labels, uncond_labels, guidance_scale, sigma_data, n):
    """Run a single CFG denoiser call at noise level sigma.

    Returns:
        D: denoised image estimate  D(x, σ, c) = c_skip*x + c_out*F_cfg
        d: ODE direction            d = (x - D) / σ
    """
    s2 = float(sigma) ** 2
    d2 = float(sigma_data) ** 2
    c_skip  = d2 / (s2 + d2)
    c_out   = float(sigma) * float(sigma_data) / np.sqrt(s2 + d2)
    c_in    = 1.0 / np.sqrt(s2 + d2)
    c_noise = float(np.log(sigma) / 4.0)

    x_in = tf.cast(c_in * x, tf.float32)
    t_in = tf.fill([n], tf.constant(c_noise, dtype=tf.float32))

    # CFG: single concatenated forward pass to halve kernel launches
    x_cat = tf.concat([x_in, x_in], axis=0)
    t_cat = tf.concat([t_in, t_in], axis=0)
    l_cat = tf.concat([uncond_labels, cond_labels], axis=0)
    F_both, _ = model([x_cat, t_cat, l_cat], training=False)
    F_both = tf.cast(F_both, tf.float32)
    # uncond is first half, cond is second half (matches order in dataset CFG dropout)
    F_cfg = F_both[:n] + guidance_scale * (F_both[n:] - F_both[:n])

    D = tf.constant(c_skip, dtype=tf.float32) * x + tf.constant(c_out, dtype=tf.float32) * F_cfg
    d = (x - D) / tf.constant(float(sigma), dtype=tf.float32)
    return D, d


def sample_edm_batch(model, class_labels, num_classes, sigmas, sigma_data=0.139,
                     guidance_scale=3.0, image_size=128):
    """Generate a batch of images via the EDM 2nd-order Heun ODE sampler.

    Starts from pure Gaussian noise x ~ N(0, sigma_max²·I) and denoises through
    the σ schedule to σ=0 (clean image). No seed image required.

    Args:
        model: Trained U-Net from build_unet (expects float32 timestep input).
        class_labels: numpy int32 array of length N, target class indices.
        num_classes: total number of classes (unconditional token = num_classes).
        sigmas: descending σ array from edm_sigma_schedule(), last element must be 0.
        sigma_data: std of training data (must match training config).
        guidance_scale: CFG scale. 1.0 = no guidance, 3.0–7.0 = typical range.
        image_size: spatial resolution H=W.

    Returns:
        numpy array, shape [N, H, W, 1], values in [0, 1] (or original data range
        if diffusion_utils norm constants are loaded).
    """
    n = len(class_labels)
    x = tf.random.normal([n, image_size, image_size, 1],
                         dtype=tf.float32) * float(sigmas[0])

    cond_labels   = tf.constant(class_labels, dtype=tf.int32)
    uncond_labels = tf.fill([n], tf.constant(num_classes, dtype=tf.int32))

    for i in range(len(sigmas) - 1):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        # 1st eval: ODE direction at (x, σ_i)
        _, d_i = _denoise(model, x, sigma, cond_labels, uncond_labels,
                          guidance_scale, sigma_data, n)

        # Euler predictor
        x_euler = x + (sigma_next - sigma) * d_i

        if sigma_next > 0.0:
            # 2nd eval: Heun corrector at (x̂, σ_{i+1})
            _, d_next = _denoise(model, x_euler, sigma_next, cond_labels, uncond_labels,
                                 guidance_scale, sigma_data, n)
            x = x + (sigma_next - sigma) * (d_i + d_next) / 2.0
        else:
            # Final step σ → 0: Euler only (corrector would divide by zero)
            x = x_euler

    x = tf.clip_by_value(x, 0.0, 1.0)
    if diffusion_utils.DATA_MIN is not None:
        x = diffusion_utils.denormalize(x)
    return x.numpy()


def load_model(config, checkpoint_path):
    """Load a trained EDM U-Net and its normalization constants.

    Unlike the DDPM load_model, this does NOT call init_schedule — EDM sampling
    uses sigma schedules directly, not the cosine/linear alpha-cumprod tables.
    """
    from diffusion.diffusion_model import build_unet

    print(f"\nLoading model from: {checkpoint_path}")

    norm_path = os.path.join(config['checkpoint_dir'], 'norm_constants.json')
    if os.path.exists(norm_path):
        diffusion_utils.load_norm_constants(norm_path)
        print(f"  Normalization constants: min={diffusion_utils.DATA_MIN:.6f}, max={diffusion_utils.DATA_MAX:.6f}")
    else:
        print("  Warning: norm_constants.json not found; samples will be in [0, 1].")

    model = build_unet(config)
    model.load_weights(checkpoint_path)
    print(f"  Parameters: {model.count_params():,}")
    print("Model loaded successfully")
    return model


def generate_dataset_edm(model, config, samples_per_class=100, guidance_scale=3.0,
                          num_steps=40, output_dir=None):
    """Generate a synthetic dataset for all classes using the EDM ODE sampler.

    Produces the same .npy file format as diffusion_sample.generate_dataset so that
    evaluation/evaluation.py evaluate_diffusion_map() can consume the output without
    any changes.

    Each class is written to a per-class temp file immediately after generation so
    that a crash or timeout only loses the class currently in progress. If temp files
    from a previous run exist, those classes are skipped (resume support). Temp files
    are deleted once the final merged output is saved successfully.

    Args:
        model: Trained U-Net loaded via load_model().
        config: Config dict (get_config('diagnostic') etc.).
        samples_per_class: Images to generate per class.
        guidance_scale: CFG scale w.
        num_steps: Number of ODE steps (40 is the EDM paper default).
        output_dir: Where to save the .npy files. Defaults to config['sample_dir'].

    Returns:
        X_synthetic: [N, H, W] float32 numpy array.
        y_synthetic: [N, num_classes] float32 one-hot numpy array.
    """
    import shutil

    num_classes = config['num_classes']
    image_size  = config['image_size']
    sigma_data  = config.get('sigma_data', 0.139)
    sigma_max   = config.get('sigma_max', 80.0)
    sigma_min   = config.get('sigma_min', 0.002)
    rho         = config.get('sigma_rho', 7)

    sigmas = diffusion_utils.edm_sigma_schedule(sigma_max, sigma_min, num_steps, rho)

    out_dir = output_dir or config['sample_dir']
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, f'_tmp_w{guidance_scale:.1f}')
    os.makedirs(tmp_dir, exist_ok=True)

    def _tmp_path(class_id):
        return os.path.join(tmp_dir, f'class_{class_id:03d}.npy')

    already_done = [c for c in range(num_classes) if os.path.exists(_tmp_path(c))]
    if already_done:
        print(f"\nResuming: {len(already_done)}/{num_classes} classes already saved in {tmp_dir}")

    print(f"\nGenerating synthetic dataset (EDM Heun sampler):")
    print(f"  Classes:           {num_classes}")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Total samples:     {num_classes * samples_per_class}")
    print(f"  Guidance scale:    {guidance_scale}")
    print(f"  ODE steps:         {num_steps}  (2×{num_steps - 1} + 1 model evals)")
    print(f"  Per-class temp dir: {tmp_dir}")

    for class_id in tqdm(range(num_classes), desc='Classes'):
        tmp_path = _tmp_path(class_id)
        if os.path.exists(tmp_path):
            continue  # already generated in a previous run

        class_labels = np.full(samples_per_class, class_id, dtype=np.int32)
        samples = sample_edm_batch(
            model,
            class_labels=class_labels,
            num_classes=num_classes,
            sigmas=sigmas,
            sigma_data=sigma_data,
            guidance_scale=guidance_scale,
            image_size=image_size,
        )  # [N, H, W, 1]

        labels = np.zeros((samples_per_class, num_classes), dtype=np.float32)
        labels[:, class_id] = 1.0

        # Pack labels + flattened images into a single array so the class is atomic
        imgs_flat = samples[..., 0].reshape(samples_per_class, -1)   # [N, H*W]
        payload   = np.concatenate([labels, imgs_flat], axis=1).astype(np.float32)
        np.save(tmp_path, payload)

    # Merge all per-class files into the final output arrays
    print("\nMerging per-class files...")
    all_images, all_labels = [], []
    for class_id in range(num_classes):
        payload   = np.load(_tmp_path(class_id))
        lbl       = payload[:, :num_classes]
        imgs_flat = payload[:, num_classes:]
        all_labels.append(lbl)
        all_images.append(imgs_flat.reshape(samples_per_class, image_size, image_size))

    X_synthetic = np.concatenate(all_images, axis=0)
    y_synthetic = np.concatenate(all_labels, axis=0)

    indices = np.arange(len(X_synthetic))
    np.random.shuffle(indices)
    X_synthetic = X_synthetic[indices]
    y_synthetic = y_synthetic[indices]

    feat_path = os.path.join(out_dir, f'diffusion_synthetic_expressions_w{guidance_scale:.1f}.npy')
    lbl_path  = os.path.join(out_dir, f'diffusion_synthetic_labels_w{guidance_scale:.1f}.npy')
    np.save(feat_path, X_synthetic)
    np.save(lbl_path,  y_synthetic)

    print(f"\nSaved synthetic dataset:")
    print(f"  Features: {feat_path}  {X_synthetic.shape}")
    print(f"  Labels:   {lbl_path}  {y_synthetic.shape}")
    print(f"  Value range: [{X_synthetic.min():.4f}, {X_synthetic.max():.4f}]")

    shutil.rmtree(tmp_dir)
    print(f"  Temp files cleaned up.")

    return X_synthetic, y_synthetic


if __name__ == "__main__":
    import argparse
    from diffusion.diffusion_config import get_config

    parser = argparse.ArgumentParser(
        description='Generate samples from a trained EDM diffusion model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m diffusion.diffusion_edm_sample --mode diagnostic \\
      --checkpoint output/diffusion/diagnostic/checkpoints/diffusion_model_final.weights.h5 \\
      --generate-dataset --samples-per-class 100 --guidance-scale 3.0

  python -m diffusion.diffusion_edm_sample --mode remote \\
      --checkpoint output/diffusion/remote/checkpoints/diffusion_model_ema.weights.h5 \\
      --generate-dataset --guidance-scale 5.0 --num-steps 40
        """,
    )
    parser.add_argument('--mode', type=str, choices=['local', 'remote', 'diagnostic'],
                        default='diagnostic', help='Config mode (default: diagnostic)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model weights (.weights.h5)')
    parser.add_argument('--generate-dataset', action='store_true',
                        help='Generate full synthetic dataset and save .npy files')
    parser.add_argument('--samples-per-class', type=int, default=100,
                        help='Images to generate per class (default: 100)')
    parser.add_argument('--guidance-scale', type=float, default=3.0,
                        help='CFG guidance scale w (default: 3.0)')
    parser.add_argument('--num-steps', type=int, default=40,
                        help='Number of EDM ODE steps (default: 40)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for .npy files (default: config sample_dir)')

    args = parser.parse_args()
    config = get_config(args.mode)
    model  = load_model(config, args.checkpoint)

    if args.generate_dataset:
        generate_dataset_edm(
            model, config,
            samples_per_class=args.samples_per_class,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
        )
    else:
        print("\nNo action specified. Use --generate-dataset.")
        print("Run with --help for usage examples.")
