import os
import json

import numpy as np
import tensorflow as tf

# Global normalization constants (computed from full dataset)
DATA_MIN = None
DATA_MAX = None

# Global variance schedule state (populated by init_schedule)
T = None
alpha = None
alpha_cumprod = None
beta = None


def init_schedule(num_timesteps, kind='cosine'):
    """Initialize the global variance schedule for diffusion.

    Args:
        num_timesteps: Number of diffusion timesteps T.
        kind: 'cosine' (default) or 'linear'.

    Must be called before any prepare_batch_* / sampling functions.
    """
    global T, alpha, alpha_cumprod, beta
    T = num_timesteps
    if kind == 'linear':
        alpha, alpha_cumprod, beta = linear_variance_schedule(num_timesteps)
    elif kind == 'cosine':
        alpha, alpha_cumprod, beta = variance_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown variance schedule: {kind}. Use 'cosine' or 'linear'.")
    return alpha, alpha_cumprod, beta


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


def variance_schedule(T, s=0.008, max_beta=0.999):
    """Cosine variance schedule (Nichol & Dhariwal, 2021)."""
    t = np.arange(T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha = np.clip(f[1:] / f[:-1], 1 - max_beta, 1)
    alpha = np.append(1, alpha).astype(np.float32)  # add α₀ = 1
    beta = 1 - alpha
    alpha_cumprod = np.cumprod(alpha)
    return alpha, alpha_cumprod, beta  # αₜ , α̅ₜ , βₜ for t = 0 to T


def linear_variance_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule (Ho et al., 2020 DDPM).

    Spreads noise more uniformly across timesteps than cosine — preferable for
    sparse data where cosine over-preserves structure at small t.
    """
    beta_inner = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alpha_inner = 1.0 - beta_inner
    # Prepend dummy step 0 (α₀=1, β₀=0) so indexing matches cosine schedule
    alpha = np.concatenate([[1.0], alpha_inner]).astype(np.float32)
    beta = np.concatenate([[0.0], beta_inner]).astype(np.float32)
    alpha_cumprod = np.cumprod(alpha)
    return alpha, alpha_cumprod, beta

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


def prepare_batch(X):
    """Prepare batch for diffusion training with global normalization.
    
    Args:
        X: Images, shape (N, H, W)
        
    Returns:
        Dictionary with X_noisy and time, plus noise tensor
    """
    X = tf.cast(X[..., tf.newaxis], tf.float32)
    
    # Normalize to [0, 1] using global min/max
    if DATA_MIN is not None and DATA_MAX is not None:
        X = (X - DATA_MIN) / (DATA_MAX - DATA_MIN)  # → [0, 1]
    
    X_shape = tf.shape(X)
    t = tf.random.uniform([X_shape[0]], minval=1, maxval=T + 1, dtype=tf.int32)
    alpha_cm = tf.gather(alpha_cumprod, t)
    alpha_cm = tf.reshape(alpha_cm, [X_shape[0]] + [1] * (len(X_shape) - 1))
    noise = tf.random.normal(X_shape)
    return {
        "X_noisy": alpha_cm ** 0.5 * X + (1 - alpha_cm) ** 0.5 * noise,
        "time": t,
    }, noise

def prepare_dataset(X, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(X)
    if shuffle:
        ds = ds.shuffle(10_000)
    return ds.batch(batch_size).map(prepare_batch).prefetch(1)


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


def prepare_batch_conditional(X, y, num_classes, dropout_rate=0.15, max_timestep=None, min_timestep=1):
    """
    Prepare batch for conditional diffusion training with classifier-free guidance.

    Args:
        X: Images, shape (N, H, W)
        y: One-hot encoded labels, shape (N, num_classes)
        num_classes: Total number of classes
        dropout_rate: Classifier-free guidance dropout rate
        max_timestep: Upper bound for sampled timesteps (inclusive). Defaults to T.
        min_timestep: Lower bound for sampled timesteps (inclusive). Defaults to 1.
            Set both to restrict training to a specific noise band [min, max].

    Returns:
        Tuple of (inputs_dict, true_noise) where:
        - inputs_dict contains X_noisy, timesteps, class_labels
        - true_noise is the noise that was added
    """
    if T is None or alpha_cumprod is None:
        raise RuntimeError(
            "Diffusion schedule is not initialized. Call diffusion_utils.init_schedule(T) first."
        )

    # Normalize images (accept both (N, H, W) and (N, H, W, 1) inputs)
    X = tf.cast(X, tf.float32)
    if X.shape.rank == 3:
        X = X[..., tf.newaxis]
    X = forward_transform(X)

    # Sample random timesteps within [min_timestep, max_timestep]
    X_shape = tf.shape(X)
    batch_size = X_shape[0]
    t_max = max_timestep if max_timestep is not None else T
    t_min = min_timestep if min_timestep is not None else 1
    t = tf.random.uniform([batch_size], minval=t_min, maxval=t_max + 1, dtype=tf.int32)
    
    # Forward diffusion process
    alpha_cm = tf.gather(alpha_cumprod, t)
    alpha_cm = tf.reshape(alpha_cm, [batch_size] + [1] * (len(X_shape) - 1))
    noise = tf.random.normal(X_shape)
    X_noisy = alpha_cm ** 0.5 * X + (1 - alpha_cm) ** 0.5 * noise
    
    # Convert one-hot labels to class indices
    class_labels = tf.argmax(y, axis=-1, output_type=tf.int32)
    
    # Apply classifier-free dropout
    class_labels = apply_classifier_free_dropout(class_labels, num_classes, dropout_rate)
    
    return {
        'X_noisy': X_noisy,
        'timesteps': t,
        'class_labels': class_labels
    }, noise


def prepare_dataset_conditional(X, y, num_classes, batch_size=32, dropout_rate=0.15,
                                 shuffle=True, drop_remainder=False, excluded_classes=None,
                                 timestep_range=None):
    """
    Create dataset for conditional diffusion training.

    Args:
        X: Images array, shape (N, H, W)
        y: One-hot labels array, shape (N, num_classes)
        num_classes: Total number of classes
        batch_size: Batch size
        dropout_rate: Classifier-free guidance dropout rate
        shuffle: Whether to shuffle the dataset
        excluded_classes: Optional list of class indices to exclude from training.
        timestep_range: Optional [t_min, t_max] pair restricting sampled timesteps.
            Defaults to [1, T] (full schedule). Matches noise_timestep_range in config.

    Returns:
        tf.data.Dataset yielding (inputs_dict, true_noise)
    """
    if excluded_classes:
        class_indices = np.argmax(y, axis=-1)
        keep = np.ones(len(X), dtype=bool)
        for c in excluded_classes:
            keep &= (class_indices != c)
        X = X[keep]
        y = y[keep]
        print(f"  Excluded classes {excluded_classes}: {keep.sum():,} / {len(keep):,} samples retained")

    t_min = timestep_range[0] if timestep_range is not None else 1
    t_max = timestep_range[1] if timestep_range is not None else None

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(10_000)

    def map_fn(x, y):
        return prepare_batch_conditional(x, y, num_classes, dropout_rate, t_max, t_min)

    return ds.batch(batch_size, drop_remainder=drop_remainder).map(
        lambda x, y: map_fn(x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


def prepare_batch_ordinal(X, timesteps):
    """
    Add noise to images at specific timesteps (ordinal, not random).
    
    Args:
        X: Images, shape (N, H, W) or (N, H, W, C)
        timesteps: Array of timesteps for each image, shape (N,)
        
    Returns:
        Noisy images and the noise that was added
    """
    if alpha_cumprod is None:
        raise RuntimeError(
            "Diffusion schedule is not initialized. Call diffusion_utils.init_schedule(T) first."
        )

    # Add channel dimension if needed
    if len(X.shape) == 3:
        X = X[..., np.newaxis]
    X = tf.cast(X, tf.float32)
    X = forward_transform(X)

    X_shape = tf.shape(X)
    t = tf.cast(timesteps, tf.int32)
    
    # Get alpha values for these timesteps
    alpha_cm = tf.gather(alpha_cumprod, t)
    alpha_cm = tf.reshape(alpha_cm, [X_shape[0]] + [1] * (len(X_shape) - 1))
    
    # Generate noise
    noise = tf.random.normal(X_shape)
    
    # Apply forward diffusion
    X_noisy = alpha_cm ** 0.5 * X + (1 - alpha_cm) ** 0.5 * noise
    
    return X_noisy.numpy(), noise.numpy()


def visualize_diffusion_process(image, timesteps, output_path="output/preprocessing/diffusion_process.png"):
    """
    Visualize how an image degrades through the diffusion process.
    
    Args:
        image: Single image, shape (H, W) or (H, W, 1)
        timesteps: List of timesteps to visualize (e.g., [0, 500, 1000, ...])
        output_path: Where to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Ensure image is 2D for processing
    if len(image.shape) == 3:
        image = image[:, :, 0]
    
    # Prepare batch with this image repeated for each timestep
    batch = np.repeat(image[np.newaxis, ...], len(timesteps), axis=0)
    
    # Add noise at each timestep
    noisy_images, _ = prepare_batch_ordinal(batch, timesteps)
    
    # Create visualization
    n_steps = len(timesteps)
    n_cols = min(5, n_steps)  # Max 5 columns
    n_rows = (n_steps + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (t, noisy_img) in enumerate(zip(timesteps, noisy_images)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Remove channel dimension for display
        if len(noisy_img.shape) == 3:
            noisy_img = noisy_img[:, :, 0]
        
        ax.imshow(noisy_img, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f't = {t}\nα̅ = {alpha_cumprod[t]:.4f}', fontsize=10)
        ax.axis('off')
    
    # Hide any unused subplots
    for idx in range(len(timesteps), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Diffusion Process: Progressive Noise Addition', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Diffusion process visualization saved to {output_path}")


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

    np.random.seed(42)  # extra code – for reproducibility

    # Round-trip validation gate (must pass before training)
    _round_trip_test()

    T = 1000
    alpha, alpha_cumprod, beta = variance_schedule(T)
    
    # Load data
    X_train = np.load("output/preprocessing/resized_expressions.npy")
    print(f"\nLoaded expression images: {X_train.shape}")
    print(f"Data range: [{DATA_MIN:.6f}, {DATA_MAX:.6f}]")
    print(f"Normalization will map to [0, 1]")
    
    # Visualize diffusion process for first 3 samples
    timesteps = list(range(0, 1001, 100))  # [0, 100, 200, ... 1000]
    print(f"\nVisualizing diffusion at timesteps: {timesteps}")
    
    for sample_idx in [0, 100, 500]:
        print(f"\nProcessing sample {sample_idx}...")
        visualize_diffusion_process(
            X_train[sample_idx],
            timesteps,
            output_path=f"output/preprocessing/diffusion_process_sample_{sample_idx}.png"
        )
    
    # Create training dataset with random timesteps
    tf.random.set_seed(43)
    train_set = prepare_dataset(X_train, batch_size=32, shuffle=True)
    print(f"\nTraining dataset created with random timesteps")
    print(f"Total timesteps in schedule: {T}")
    print(f"Alpha range: [{alpha_cumprod.min():.6f}, {alpha_cumprod.max():.6f}]")