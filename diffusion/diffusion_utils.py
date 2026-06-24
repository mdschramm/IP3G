import os
import json

import numpy as np
import tensorflow as tf

# Global normalization constants (computed from full dataset)
DATA_MIN = None
DATA_MAX = None

def set_data_range(data_min, data_max):
    """Set the global linear data range used for [-1, 1] normalization."""
    global DATA_MIN, DATA_MAX
    DATA_MIN = float(data_min)
    DATA_MAX = float(data_max)


def save_norm_constants(path):
    """Persist DATA_MIN / DATA_MAX next to a checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, 'w') as f:
        json.dump({'data_min': DATA_MIN, 'data_max': DATA_MAX}, f)


def load_norm_constants(path):
    """Load DATA_MIN / DATA_MAX saved by save_norm_constants."""
    with open(path, 'r') as f:
        payload = json.load(f)
    set_data_range(payload['data_min'], payload['data_max'])
    return DATA_MIN, DATA_MAX


def forward_transform(X):
    """Map raw data to [0, 1] using minmax normalization.

    Accepts numpy array or TF tensor; returns a TF tensor.
    """
    X = tf.cast(X, tf.float32)
    if DATA_MIN is not None and DATA_MAX is not None:
        return (X - DATA_MIN) / (DATA_MAX - DATA_MIN)
    return X


def denormalize(x):
    """Map values from [0, 1] back to the original data range."""
    x01 = tf.cast(x, tf.float32)
    if DATA_MIN is None or DATA_MAX is None:
        return tf.clip_by_value(x01, 0.0, 1.0)
    return x01 * (DATA_MAX - DATA_MIN) + DATA_MIN


# ─────────────────────────────────────────────────────────────────────────────
# EDM2 noise parameterization (Karras et al. 2022/2023)
# ─────────────────────────────────────────────────────────────────────────────

def edm_preconditioning(sigma, sigma_data=0.139):
    """Preconditioning scalars c_skip, c_out, c_in, c_noise (numpy, any sigma shape)."""
    s2 = sigma ** 2
    d2 = sigma_data ** 2
    c_skip  = d2 / (s2 + d2)
    c_out   = sigma * sigma_data / np.sqrt(s2 + d2)
    c_in    = 1.0 / np.sqrt(s2 + d2)
    c_noise = np.log(sigma) / 4.0
    return c_skip, c_out, c_in, c_noise


def sample_sigma_lognormal(batch_size, P_mean=-2.0, P_std=1.2,
                            sigma_min=0.002, sigma_max=80.0):
    """Draw σ from truncated log-normal: ln(σ) ~ N(P_mean, P_std²)."""
    ln_sigma = np.random.normal(P_mean, P_std, size=batch_size).astype(np.float32)
    return np.clip(np.exp(ln_sigma), sigma_min, sigma_max)


def edm_forward_diffuse(x0, sigma):
    """EDM forward diffusion: x_t = x0 + σ·ε.

    Args:
        x0: normalized images [N, H, W, 1]
        sigma: per-sample noise levels, shape [N] or scalar
    Returns:
        x_t: noisy images, same shape as x0
        eps: the Gaussian noise added
    """
    eps = np.random.normal(size=x0.shape).astype(np.float32)
    sigma = np.reshape(sigma, [-1, 1, 1, 1]) if np.ndim(sigma) > 0 else float(sigma)
    return x0 + sigma * eps, eps


def edm_sigma_schedule(sigma_max=80.0, sigma_min=0.002, num_steps=40, rho=7):
    """Karras et al. σ schedule for inference.

    Returns num_steps values from sigma_max down to sigma_min, then appends 0
    so the final Euler step maps to a clean image prediction.
    Shape: [num_steps + 1].
    """
    steps = np.arange(num_steps)
    hi = sigma_max ** (1.0 / rho)
    lo = sigma_min ** (1.0 / rho)
    sigmas = ((hi + steps / max(num_steps - 1, 1) * (lo - hi)) ** rho).astype(np.float32)
    return np.append(sigmas, 0.0)


def prepare_batch_conditional_edm(X, y, num_classes, dropout_rate=0.15,
                                   P_mean=-2.0, P_std=1.2,
                                   sigma_min=0.002, sigma_max=80.0, sigma_data=0.139):
    """EDM2 training batch with preconditioning applied in the dataset.

    Samples σ from log-normal, forward-diffuses x0, applies c_in/c_skip/c_out
    preconditioning so the loss target has unit weight at every noise level.

    Returns:
        inputs_dict: X_noisy (c_in-scaled), timesteps (c_noise float32), class_labels, sigma
        target: preconditioned denoising target for F; plain MSE loss, no w(σ) needed
    """
    X = tf.cast(X, tf.float32)
    if X.shape.rank == 3:
        X = X[..., tf.newaxis]
    X = forward_transform(X)
    X_shape = tf.shape(X)
    batch_size = X_shape[0]

    # Log-normal σ sampling — tf.random for graph-mode compatibility
    ln_sigma = tf.random.normal([batch_size], mean=float(P_mean), stddev=float(P_std))
    sigma = tf.clip_by_value(tf.exp(ln_sigma), float(sigma_min), float(sigma_max))

    # EDM forward diffuse: x_t = x0 + σ·ε
    eps    = tf.random.normal(X_shape)
    sig_b  = tf.reshape(sigma, [batch_size, 1, 1, 1])
    x_t    = X + sig_b * eps

    # Preconditioning scalars
    sd  = tf.constant(float(sigma_data), dtype=tf.float32)
    s2  = sigma ** 2
    d2  = sd ** 2
    c_skip  = d2 / (s2 + d2)
    c_out   = sigma * sd / tf.sqrt(s2 + d2)
    c_in    = 1.0 / tf.sqrt(s2 + d2)
    c_noise = tf.math.log(sigma) / 4.0      # float conditioning signal for sinusoidal embedding

    c_in_b   = tf.reshape(c_in,   [batch_size, 1, 1, 1])
    c_skip_b = tf.reshape(c_skip, [batch_size, 1, 1, 1])
    c_out_b  = tf.reshape(c_out,  [batch_size, 1, 1, 1])

    x_in   = c_in_b * x_t                      # scaled model input
    target = (X - c_skip_b * x_t) / c_out_b    # unit-weight target for F

    class_labels = tf.argmax(y, axis=-1, output_type=tf.int32)
    class_labels = apply_classifier_free_dropout(class_labels, num_classes, dropout_rate)

    return {
        'X_noisy':      x_in,
        'timesteps':    c_noise,
        'class_labels': class_labels,
        'sigma':        sigma,          # passed through for diagnostic logging
    }, target


def prepare_dataset_conditional_edm(X, y, num_classes, batch_size=32, dropout_rate=0.15,
                                     shuffle=True, drop_remainder=False, excluded_classes=None,
                                     P_mean=-2.0, P_std=1.2, sigma_min=0.002, sigma_max=80.0,
                                     sigma_data=0.139):
    """Create tf.data.Dataset for EDM2 conditional training."""
    if excluded_classes:
        class_indices = np.argmax(y, axis=-1)
        keep = np.ones(len(X), dtype=bool)
        for c in excluded_classes:
            keep &= (class_indices != c)
        X = X[keep]
        y = y[keep]
        print(f"  Excluded classes {excluded_classes}: {keep.sum():,} / {len(keep):,} samples retained")

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(10_000)

    def map_fn(x, lbl):
        return prepare_batch_conditional_edm(
            x, lbl, num_classes, dropout_rate,
            P_mean, P_std, sigma_min, sigma_max, sigma_data,
        )

    return ds.batch(batch_size, drop_remainder=drop_remainder).map(
        map_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


def apply_classifier_free_dropout(class_labels, num_classes, dropout_rate=0.15):
    """
    Randomly replace class labels with unconditional token for classifier-free guidance.
    
    Args:
        class_labels: Tensor of class indices, shape [batch_size]
        num_classes: Total number of classes
        dropout_rate: Probability of replacing with unconditional token
        
    Returns:
        Modified class labels with some replaced by unconditional token (num_classes)
    """
    batch_size = tf.shape(class_labels)[0]
    mask = tf.random.uniform([batch_size]) < dropout_rate
    unconditional_token = tf.constant(num_classes, dtype=class_labels.dtype)
    return tf.where(mask, unconditional_token, class_labels)


def _round_trip_test(data_path="output/preprocessing/resized_expressions.npy"):
    """Validate that forward_transform → denormalize is the identity (within fp32 precision)."""
    print("\nRunning normalization round-trip test...")
    X_real = np.load(data_path).astype(np.float32)
    set_data_range(X_real.min(), X_real.max())
    print(f"  DATA_MIN/MAX: [{DATA_MIN:.6f}, {DATA_MAX:.6f}]")

    X_norm = forward_transform(X_real).numpy()
    X_recovered = denormalize(tf.constant(X_norm)).numpy()

    nmin, nmax = float(X_norm.min()), float(X_norm.max())
    err = float(np.abs(X_real - X_recovered).max())
    print(f"  Normalized range: [{nmin:.6f}, {nmax:.6f}]")
    print(f"  Max round-trip error: {err:.2e}")

    assert nmin >= 0.0 - 1e-5 and nmax <= 1.0 + 1e-5, f"normalized out of [0,1]: [{nmin}, {nmax}]"
    assert err < 1e-5, f"Round-trip failed: max err = {err:.2e}"
    print("  Round-trip clean")


if __name__ == "__main__":
    # Round-trip validation: forward_transform → denormalize should be identity.
    _round_trip_test()