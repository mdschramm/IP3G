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

import numpy as np
import tensorflow as tf

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
