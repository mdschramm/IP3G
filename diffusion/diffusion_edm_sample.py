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
    mask    = tf.constant(np.load(DEFAULT_CONFIG.pixel_occupancy_mask_path).astype("float32"))
    samples = sample_edm_batch(
        model,
        class_labels=np.array([0, 1, 5, 12]),
        num_classes=54,
        sigmas=sigmas,
        sigma_data=0.139,
        guidance_scale=3.0,
        image_size=128,
        occupancy_mask=mask,
    )  # → numpy [N, H, W, 16] in [0, 1]
"""

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import diffusion.diffusion_utils as diffusion_utils


def _denoise(model, x, sigma, cond_labels, uncond_labels, guidance_scale, sigma_data, n,
             mask_batch):
    """Run a single CFG denoiser call at noise level sigma.

    mask_batch: (n, H, W, C) float32 occupancy mask — broadcast to 2n for the
                concatenated CFG forward pass.

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
    x_cat    = tf.concat([x_in, x_in], axis=0)
    t_cat    = tf.concat([t_in, t_in], axis=0)
    l_cat    = tf.concat([uncond_labels, cond_labels], axis=0)
    mask_cat = tf.concat([mask_batch, mask_batch], axis=0)
    F_both, _ = model([x_cat, t_cat, l_cat, mask_cat], training=False)
    F_both = tf.cast(F_both, tf.float32)
    # uncond is first half, cond is second half (matches order in dataset CFG dropout)
    F_cfg = F_both[:n] + guidance_scale * (F_both[n:] - F_both[:n])

    D = tf.constant(c_skip, dtype=tf.float32) * x + tf.constant(c_out, dtype=tf.float32) * F_cfg
    d = (x - D) / tf.constant(float(sigma), dtype=tf.float32)
    return D, d


def sample_edm_batch(model, class_labels, num_classes, sigmas, sigma_data=0.139,
                     guidance_scale=3.0, image_size=128, occupancy_mask=None,
                     null_tokens=None, uncond_labels=None):
    """Generate a batch of images via the EDM 2nd-order Heun ODE sampler.

    Starts from pure Gaussian noise x ~ N(0, sigma_max²·I) masked to occupied
    channels only, then denoises through the σ schedule to σ=0. The occupancy
    mask is applied after every ODE step to keep structural zeros exactly zero.

    Args:
        model: Trained U-Net from build_unet (expects float32 timestep input).
        class_labels: numpy int32 array of length N, target class indices.
        num_classes: total number of classes (unconditional token = num_classes).
        sigmas: descending σ array from edm_sigma_schedule(), last element must be 0.
        sigma_data: std of occupied training positions (must match training config).
        guidance_scale: CFG scale. 1.0 = no guidance, 3.0–7.0 = typical range.
        image_size: spatial resolution H=W.
        occupancy_mask: (H, W, C) float32 tensor or None. If None a dense all-ones
                        mask is used (backward compatible with single-channel models).
        null_tokens: FACTORIZED models only — per-attribute null token ids, i.e.
                     [vocab_size for each attribute]. class_labels is then [N, A]
                     and the unconditional branch is the all-null row.
        uncond_labels: the negative side of the guidance pair, overriding the
                     all-null default. This is what makes guidance per-attribute:
                     pass a row that nulls ONLY the attribute being guided and
                     keeps the others at their conditional values, and the
                     (cond - uncond) difference isolates that one attribute's
                     direction instead of the whole conditioning vector's.

    Returns:
        numpy array, shape [N, H, W, C], values in [0, 1].
    """
    n          = len(class_labels)
    in_channels = model.input_shape[0][-1]  # derive C from model input spec

    if occupancy_mask is not None:
        mask = tf.cast(occupancy_mask, tf.float32)           # (H, W, C)
    else:
        mask = tf.ones([image_size, image_size, in_channels], dtype=tf.float32)

    mask_batch = tf.tile(mask[None], [n, 1, 1, 1])          # (n, H, W, C)

    # Start from noise in occupied channels only; structural zeros stay 0
    x = tf.random.normal([n, image_size, image_size, in_channels],
                         dtype=tf.float32) * float(sigmas[0]) * mask_batch

    cond_labels = tf.constant(class_labels, dtype=tf.int32)
    if uncond_labels is not None:
        uncond_labels = tf.constant(uncond_labels, dtype=tf.int32)
    elif cond_labels.shape.rank == 2:
        if null_tokens is None:
            raise ValueError(
                "Factorized class_labels ([N, A]) need null_tokens=[vocab_size per "
                "attribute] so the unconditional branch can be built."
            )
        if len(null_tokens) != cond_labels.shape[1]:
            raise ValueError(
                f"null_tokens has {len(null_tokens)} entries but class_labels has "
                f"{cond_labels.shape[1]} attribute columns."
            )
        uncond_labels = tf.broadcast_to(
            tf.constant(list(null_tokens), dtype=tf.int32)[None, :], tf.shape(cond_labels))
    else:
        uncond_labels = tf.fill([n], tf.constant(num_classes, dtype=tf.int32))

    for i in range(len(sigmas) - 1):
        sigma      = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        # 1st eval: ODE direction at (x, σ_i)
        _, d_i = _denoise(model, x, sigma, cond_labels, uncond_labels,
                          guidance_scale, sigma_data, n, mask_batch)

        # Euler predictor
        x_euler = x + (sigma_next - sigma) * d_i
        x_euler = x_euler * mask_batch   # enforce structural zeros after Euler step

        if sigma_next > 0.0:
            # 2nd eval: Heun corrector at (x̂, σ_{i+1})
            _, d_next = _denoise(model, x_euler, sigma_next, cond_labels, uncond_labels,
                                 guidance_scale, sigma_data, n, mask_batch)
            x = x + (sigma_next - sigma) * (d_i + d_next) / 2.0
        else:
            # Final step σ → 0: Euler only (corrector would divide by zero)
            x = x_euler

        x = x * mask_batch   # re-enforce after each step

    x = tf.clip_by_value(x, 0.0, 1.0)
    x = x * mask_batch   # final re-enforcement after clip
    if diffusion_utils.DATA_MIN is not None:
        x = diffusion_utils.denormalize(x)
    return x.numpy()


def load_model(config, checkpoint_path):
    """Load a trained EDM U-Net and its normalization constants.

    Unlike the DDPM load_model, this does NOT call init_schedule — EDM sampling
    uses sigma schedules directly, not the cosine/linear alpha-cumprod tables.
    """
    import keras
    from diffusion.diffusion_model import build_unet

    print(f"\nLoading model from: {checkpoint_path}")

    if config.get('mixed_precision', False):
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("  Mixed precision: float16 compute / float32 variables")

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
                          num_steps=40, output_dir=None, classes_per_batch=1):
    """Generate a synthetic dataset for all classes using the EDM ODE sampler.

    Produces .npy files compatible with evaluation/evaluation.py evaluate_diffusion_map().

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
        classes_per_batch: Number of classes to generate simultaneously. Each class
            contributes samples_per_class samples; the CFG pass sees
            2 * classes_per_batch * samples_per_class images per forward call.
            Use 1 on M1 (shared memory), 4–8 on T4 (16 GB GDDR6).

    Returns:
        X_synthetic: [N, H, W, C] float32 numpy array.
        y_synthetic: [N, num_classes] float32 one-hot numpy array.
    """
    import shutil

    num_classes = config['num_classes']
    image_size  = config['image_size']
    in_channels = config['in_channels']
    sigma_data  = config.get('sigma_data', 0.139)
    sigma_max   = config.get('sigma_max', 80.0)
    sigma_min   = config.get('sigma_min', 0.002)
    rho         = config.get('sigma_rho', 7)

    sigmas = diffusion_utils.edm_sigma_schedule(sigma_max, sigma_min, num_steps, rho)

    # Load occupancy mask
    mask_path = os.path.join(config['data_dir'], 'pixel_occupancy_mask.npy')
    occupancy_mask = None
    if os.path.exists(mask_path):
        occupancy_mask = tf.constant(np.load(mask_path).astype(np.float32))
        print(f"  Occupancy mask loaded: {occupancy_mask.shape}")
    else:
        print("  Warning: pixel_occupancy_mask.npy not found; using dense mask.")

    out_dir = output_dir or config['sample_dir']
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, f'_tmp_w{guidance_scale:.1f}')
    os.makedirs(tmp_dir, exist_ok=True)

    def _tmp_path(class_id):
        return os.path.join(tmp_dir, f'class_{class_id:03d}.npy')

    already_done = [c for c in range(num_classes) if os.path.exists(_tmp_path(c))]
    if already_done:
        print(f"\nResuming: {len(already_done)}/{num_classes} classes already saved in {tmp_dir}")

    cfg_batch = 2 * classes_per_batch * samples_per_class
    print(f"\nGenerating synthetic dataset (EDM Heun sampler):")
    print(f"  Classes:           {num_classes}")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Total samples:     {num_classes * samples_per_class}")
    print(f"  Guidance scale:    {guidance_scale}")
    print(f"  ODE steps:         {num_steps}  (2×{num_steps - 1} + 1 model evals)")
    print(f"  Classes per batch: {classes_per_batch}  (CFG batch size: {cfg_batch})")
    print(f"  Per-class temp dir: {tmp_dir}")

    chunks = [list(range(num_classes))[i:i + classes_per_batch]
              for i in range(0, num_classes, classes_per_batch)]

    for chunk in tqdm(chunks, desc='Batches'):
        pending = [c for c in chunk if not os.path.exists(_tmp_path(c))]
        if not pending:
            continue

        class_labels = np.concatenate([
            np.full(samples_per_class, c, dtype=np.int32) for c in pending
        ])
        samples = sample_edm_batch(
            model,
            class_labels=class_labels,
            num_classes=num_classes,
            sigmas=sigmas,
            sigma_data=sigma_data,
            guidance_scale=guidance_scale,
            image_size=image_size,
            occupancy_mask=occupancy_mask,
        )  # [len(pending)*samples_per_class, H, W, C]

        for i, class_id in enumerate(pending):
            class_samples = samples[i * samples_per_class:(i + 1) * samples_per_class]
            labels = np.zeros((samples_per_class, num_classes), dtype=np.float32)
            labels[:, class_id] = 1.0
            # Flatten all channels: [N, H*W*C]
            imgs_flat = class_samples.reshape(samples_per_class, -1)
            payload   = np.concatenate([labels, imgs_flat], axis=1).astype(np.float32)
            np.save(_tmp_path(class_id), payload)

    # Merge all per-class files into the final output arrays
    print("\nMerging per-class files...")
    all_images, all_labels = [], []
    for class_id in range(num_classes):
        payload   = np.load(_tmp_path(class_id))
        lbl       = payload[:, :num_classes]
        imgs_flat = payload[:, num_classes:]
        all_labels.append(lbl)
        all_images.append(imgs_flat.reshape(samples_per_class, image_size, image_size, in_channels))

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

  python -m diffusion.diffusion_edm_sample --mode local \\
      --checkpoint output/diffusion/local/checkpoints/diffusion_model_final.weights.h5 \\
      --generate-dataset --guidance-scale 5.0 --num-steps 40
        """,
    )
    parser.add_argument('--mode', type=str, choices=['local', 'diagnostic'],
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
    parser.add_argument('--classes-per-batch', type=int, default=1,
                        help='Classes to generate simultaneously (default: 1). '
                             'Use 4–8 on T4 (16 GB), 1–2 on M1 (shared memory).')

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
            classes_per_batch=args.classes_per_batch,
        )
    else:
        print("\nNo action specified. Use --generate-dataset.")
        print("Run with --help for usage examples.")
