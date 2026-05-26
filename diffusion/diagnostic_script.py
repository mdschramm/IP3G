# save as diffusion/diagnose.py and run locally after syncing checkpoint
import numpy as np
import tensorflow as tf
from diffusion.diffusion_config import get_config
from diffusion.diffusion_model import build_unet
import diffusion.diffusion_utils as diffusion_utils

MODE = 'remote'
CHECKPOINT = 'output/diffusion/remote/checkpoints/diffusion_model_step_040000.weights.h5'
N_BATCHES = 200  # ~3200 samples across t values

config = get_config(MODE)
X = np.load(f"output/preprocessing/{config['feature_file']}").astype(np.float32)
y = np.load(f"output/preprocessing/{config['label_file']}").astype(np.float32)

diffusion_utils.set_data_range(X.min(), X.max())
diffusion_utils.init_schedule(config['timesteps'], kind=config['variance_schedule'])

model = build_unet(config)
model.load_weights(CHECKPOINT)

alpha_bar = tf.constant(diffusion_utils.alpha_cumprod, dtype=tf.float32)
buckets = {'t<200': [], '200-500': [], '500-800': [], 't>800': []}
class_losses = {i: [] for i in range(config['num_classes'])}

rng = np.random.default_rng(42)
for _ in range(N_BATCHES):
    idx = rng.integers(0, len(X), config['batch_size'])
    x0 = diffusion_utils.forward_transform(X[idx]).numpy()[..., np.newaxis]  # (B, 128, 128, 1)
    t = rng.integers(1, config['timesteps'] + 1, config['batch_size'])
    eps = rng.standard_normal(x0.shape).astype(np.float32)
    ab = alpha_bar.numpy()[t][:, None, None, None]
    x_noisy = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps

    pred = model([x_noisy, t.astype(np.int32),
                  np.argmax(y[idx], axis=1).astype(np.int32)], training=False)
    per_sample = np.mean((pred.numpy() - eps) ** 2, axis=(1, 2, 3))

    for i, (ti, loss_i) in enumerate(zip(t, per_sample)):
        if ti < 200:   buckets['t<200'].append(loss_i)
        elif ti < 500: buckets['200-500'].append(loss_i)
        elif ti < 800: buckets['500-800'].append(loss_i)
        else:          buckets['t>800'].append(loss_i)
        class_losses[np.argmax(y[idx[i]])].append(loss_i)

print("\n=== Loss by timestep bucket ===")
for k, v in buckets.items():
    print(f"  {k:12s}: {np.mean(v):.4f} ± {np.std(v):.4f}  (n={len(v)})")

print("\n=== Top 10 hardest classes ===")
class_means = {c: np.mean(v) for c, v in class_losses.items() if v}
for c, m in sorted(class_means.items(), key=lambda x: -x[1])[:10]:
    print(f"  Class {c:2d}: {m:.4f}")
