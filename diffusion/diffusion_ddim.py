"""
DDIM Sampler for Conditional Diffusion with Partial-Noise Seeding.

Implements DDIM (Song et al. 2020) on top of the trained U-Net. Unlike DDPM,
DDIM's non-Markovian reverse process:
  - Is deterministic when eta=0 (same seed → same output)
  - Supports arbitrary timestep subsequences (e.g. 20 steps instead of 200)
  - Can start from any x_t, not just pure Gaussian noise

Designed for use with noise_timestep_range training: generation seeds from a real
image forward-noised to t_start (= range[1]) and denoises down to t_end (= range[0]),
ensuring the sampling distribution matches the training distribution exactly.

USAGE:
    from diffusion.diffusion_ddim import sample_ddim_batch, sample_ddim
    from diffusion.diffusion_sample import load_model

    config = get_config('local')
    model = load_model(config, checkpoint_path)

    # Class-conditional editing: take a real image, denoise as a different class
    seed_images = X_train[[0, 1, 2]]   # shape [3, 128, 128]
    ntr = config['noise_timestep_range']  # [t_min, t_max]
    samples = sample_ddim_batch(
        model, seed_images,
        class_labels=[5, 12, 20],
        num_classes=config['num_classes'],
        t_start=ntr[1],    # upper bound
        t_end=ntr[0],      # lower bound
        num_steps=40,
        eta=0.0,           # deterministic
        guidance_scale=3.0,
    )  # → shape [3, 128, 128, 1] in [0, 1]

    # Reconstruction check (same class): verifies model can round-trip
    recon = sample_ddim_batch(
        model, seed_images,
        class_labels=original_class_ids,
        ...
    )
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import diffusion.diffusion_utils as diffusion_utils


# ─────────────────────────────────────────────────────────────────────────────
# Timestep sequence helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_timestep_sequence(t_start, num_steps=None, timesteps=None, t_end=1):
    """Build the list of (t, t_prev) pairs for the DDIM reverse chain.

    Args:
        t_start: Starting timestep (inclusive). Must be ≤ noise_timestep_range[1].
        num_steps: Number of denoising steps. Evenly spaces t_start..t_end.
            Ignored if `timesteps` is provided.
        timesteps: Explicit list of timesteps in descending order, e.g.
            [200, 150, 100, 50, 1]. Overrides num_steps.
        t_end: Lower bound of the denoising sequence (inclusive). Defaults to 1.
            The final pair always has t_prev=0 (alpha_cumprod[0]=1.0), so the
            terminal DDIM step produces a clean image prediction from x_{t_end}.

    Returns:
        List of (t, t_prev) int pairs. The final pair always has t_prev=0,
        which maps to alpha_cumprod[0]=1.0 (clean image terminal state).
    """
    if timesteps is not None:
        seq = [int(t) for t in timesteps]
    elif num_steps is not None:
        seq = np.linspace(t_start, t_end, num_steps, dtype=int).tolist()
    else:
        raise ValueError("Either num_steps or timesteps must be provided.")
    return list(zip(seq, seq[1:] + [0]))


# ─────────────────────────────────────────────────────────────────────────────
# Forward-diffusion seeding
# ─────────────────────────────────────────────────────────────────────────────

def noise_to_timestep(x0, t_start):
    """Forward-diffuse a clean image to timestep t_start.

    Args:
        x0: Normalized image(s) in [0, 1], shape (H, W) or (N, H, W) or (N, H, W, 1).
        t_start: Timestep to noise to.

    Returns:
        x_t: Noisy image(s), same shape as x0 with channel dim added if needed.
        eps: The Gaussian noise that was added (useful for reconstruction comparison).
    """
    if diffusion_utils.alpha_cumprod is None:
        raise RuntimeError("Call diffusion_utils.init_schedule() before noise_to_timestep.")
    x0 = np.asarray(x0, dtype=np.float32)
    if x0.ndim == 2:
        x0 = x0[np.newaxis, ..., np.newaxis]  # [1, H, W, 1]
    elif x0.ndim == 3:
        x0 = x0[..., np.newaxis]              # [N, H, W, 1]

    ab = float(diffusion_utils.alpha_cumprod[t_start])
    eps = np.random.normal(size=x0.shape).astype(np.float32)
    x_t = np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps
    return x_t, eps


# ─────────────────────────────────────────────────────────────────────────────
# Single DDIM reverse step
# ─────────────────────────────────────────────────────────────────────────────

@tf.function(reduce_retracing=True)
def _ddim_step(
    model,
    x,
    t_tensor,
    t_prev_ab,          # scalar: alpha_cumprod[t_prev] (pre-looked-up to avoid indexing inside tf.function)
    cond_labels,
    uncond_labels,
    guidance_scale,
    eta,
    eps_threshold,
):
    """Single DDIM reverse step: x_t → x_{t_prev}.

    Args:
        t_tensor: Tensor of shape [batch], all equal to current t.
        t_prev_ab: Scalar tf.float32 = alpha_cumprod[t_prev].
            alpha_cumprod[0] = 1.0, which makes the last step collapse to x0_pred.
        eta: Stochasticity weight. 0 = deterministic DDIM, 1 ≈ DDPM.
    """
    batch = tf.shape(x)[0]

    # CFG: run cond + uncond in a single concatenated forward pass
    x_in      = tf.concat([x, x], axis=0)
    t_in      = tf.concat([t_tensor, t_tensor], axis=0)
    labels_in = tf.concat([uncond_labels, cond_labels], axis=0)
    eps_both  = model([x_in, t_in, labels_in], training=False)
    eps_hat   = eps_both[:batch] + guidance_scale * (eps_both[batch:] - eps_both[:batch])

    # Optional soft-threshold on predicted noise
    eps_hat = tf.sign(eps_hat) * tf.nn.relu(tf.abs(eps_hat) - eps_threshold)

    # Alpha values for current and previous step
    ab_t      = tf.gather(tf.constant(diffusion_utils.alpha_cumprod), t_tensor[0])
    ab_t_prev = t_prev_ab   # pre-computed by caller

    # Predicted clean image (x̂₀)
    x0_pred = (x - tf.sqrt(1.0 - ab_t) * eps_hat) / tf.sqrt(ab_t)

    # DDIM variance: σ_t = η * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1})
    sigma_t = eta * tf.sqrt((1.0 - ab_t_prev) / tf.maximum(1.0 - ab_t, 1e-8)) \
                  * tf.sqrt(tf.maximum(1.0 - ab_t / tf.maximum(ab_t_prev, 1e-8), 0.0))

    # Direction pointing to x_t
    dir_xt = tf.sqrt(tf.maximum(1.0 - ab_t_prev - sigma_t ** 2, 0.0)) * eps_hat

    # x_{t_prev} = sqrt(ᾱ_{t_prev}) * x̂₀ + dir_xt + σ_t * ε
    noise = tf.random.normal(tf.shape(x))
    return tf.sqrt(ab_t_prev) * x0_pred + dir_xt + sigma_t * noise


# ─────────────────────────────────────────────────────────────────────────────
# Public sampling API
# ─────────────────────────────────────────────────────────────────────────────

def sample_ddim_batch(
    model,
    x_init,
    class_labels,
    num_classes,
    t_start,
    t_end=1,
    num_steps=50,
    timesteps=None,
    eta=0.0,
    guidance_scale=3.0,
    denormalize_output=True,
    eps_threshold=0.0,
):
    """Generate a batch of images via DDIM, seeded from real images.

    Args:
        model: Trained U-Net (from build_unet / load_model).
        x_init: Seed images in [0, 1], shape [N, H, W] or [N, H, W, 1].
            Each image is forward-noised to t_start before the reverse chain.
        class_labels: Target class indices for the reverse process, length N.
            Use the same class as x_init for reconstruction; use a different
            class for class-conditional editing.
        num_classes: Total number of classes (unconditional token = num_classes).
        t_start: Timestep to noise x_init to. Should equal noise_timestep_range[1]
            from training config. Must be ≤ the upper bound used during training.
        t_end: Lower bound of the denoising sequence. Should equal
            noise_timestep_range[0] from training config. Defaults to 1.
            The final DDIM step always produces a clean image prediction from x_{t_end}.
        num_steps: Number of DDIM denoising steps (evenly spaced t_start..t_end).
            Ignored if `timesteps` is provided. 40–50 is typically sufficient.
        timesteps: Optional explicit list [t_start, ..., t_end] in descending order.
            Overrides num_steps.
        eta: Stochasticity weight. 0.0 = deterministic DDIM (same seed → same
            output). 1.0 ≈ DDPM. Values in (0, 1) give intermediate stochasticity.
        guidance_scale: Classifier-free guidance scale.
        denormalize_output: If True, map outputs back to original data range
            via diffusion_utils.denormalize(). Otherwise return [0, 1].
        eps_threshold: Soft-threshold on predicted noise (0 = disabled).

    Returns:
        numpy array, shape [N, H, W, 1], values in [0, 1] (or original data
        range if denormalize_output=True).
    """
    class_labels = np.asarray(class_labels, dtype=np.int32)
    n = class_labels.shape[0]

    # Forward-transform seed images to [0,1] then noise to t_start
    x_init = np.asarray(x_init, dtype=np.float32)
    if x_init.ndim == 3:
        x_init = x_init[..., np.newaxis]  # [N, H, W, 1]
    x_norm = diffusion_utils.forward_transform(x_init).numpy()
    x_noisy, _ = noise_to_timestep(x_norm, t_start)
    x = tf.constant(x_noisy, dtype=tf.float32)

    # Build alpha_cumprod tensor and pre-look-up t_prev values
    alpha_cumprod = tf.constant(diffusion_utils.alpha_cumprod, dtype=tf.float32)
    step_pairs = build_timestep_sequence(t_start, num_steps=num_steps, timesteps=timesteps, t_end=t_end)

    # CFG label tensors
    cond_labels   = tf.constant(class_labels, dtype=tf.int32)
    uncond_labels = tf.fill([n], tf.constant(num_classes, dtype=tf.int32))
    guidance_scale_t = tf.constant(guidance_scale, dtype=tf.float32)
    eta_t            = tf.constant(eta, dtype=tf.float32)
    eps_threshold_t  = tf.constant(eps_threshold, dtype=tf.float32)

    for t, t_prev in tqdm(step_pairs, desc='DDIM sampling', leave=False):
        t_tensor  = tf.fill([n], tf.constant(t, dtype=tf.int32))
        t_prev_ab = alpha_cumprod[t_prev]   # scalar; alpha_cumprod[0]=1.0 at final step
        x = _ddim_step(
            model, x, t_tensor, t_prev_ab,
            cond_labels, uncond_labels,
            guidance_scale_t, eta_t, eps_threshold_t,
        )

    x = tf.clip_by_value(x, 0.0, 1.0)
    if denormalize_output:
        x = diffusion_utils.denormalize(x)
    return x.numpy()


def sample_ddim(
    model,
    x_init,
    class_label,
    num_classes,
    t_start,
    t_end=1,
    num_steps=50,
    timesteps=None,
    eta=0.0,
    guidance_scale=3.0,
    denormalize_output=True,
    eps_threshold=0.0,
):
    """Generate a single image via DDIM. Single-image wrapper around sample_ddim_batch.

    Args:
        x_init: Single seed image in [0, 1], shape [H, W] or [H, W, 1].
        class_label: Target class index (int).

    Returns:
        numpy array, shape [1, H, W, 1].
    """
    x_init = np.asarray(x_init, dtype=np.float32)
    if x_init.ndim == 2:
        x_init = x_init[np.newaxis]  # [1, H, W]
    return sample_ddim_batch(
        model,
        x_init=x_init,
        class_labels=[class_label],
        num_classes=num_classes,
        t_start=t_start,
        t_end=t_end,
        num_steps=num_steps,
        timesteps=timesteps,
        eta=eta,
        guidance_scale=guidance_scale,
        denormalize_output=denormalize_output,
        eps_threshold=eps_threshold,
    )
