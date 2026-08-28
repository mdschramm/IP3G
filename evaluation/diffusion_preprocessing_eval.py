"""
Diffusion Preprocessing Pipeline Evaluation

Visualizes sample images at each stage of the diffusion pipeline:
  1. Raw preprocessing output [0, 1]
  2. Minmax-normalized (forward_transform) — should look identical for [0,1] data
  3. Forward-noised at sigma=SIGMA_EARLY (structure still visible)
  4. Forward-noised at sigma=SIGMA_LATE (heavy noise — mostly random)

Also generates the full noise journey (sigma=0..5, ascending) via
visualize_noise_journey().

Uses the EDM2 continuous-sigma forward process (edm_forward_diffuse), matching
how the rest of the diffusion module works — there's no discrete timestep
schedule to initialize.

For multichannel configs (channels > 1), only channel 0 is displayed (the most
tissue-discriminative gene per pixel) since imshow can't render an arbitrary
channel count directly.

USAGE:
    python -m evaluation.diffusion_preprocessing_eval
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from diffusion.diffusion_utils import (
    forward_transform,
    set_data_range,
    edm_forward_diffuse,
)
from preprocessing.artifact_paths import DEFAULT_CONFIG

SAMPLE_INDICES = [0, 100, 500]
SIGMA_EARLY = 0.15   # roughly sigma_data — moderate noise, structure still visible
SIGMA_LATE = 1.5     # roughly 9x sigma_data — heavy noise, mostly random
NOISE_SIGMAS = [0.0, 0.05, SIGMA_EARLY, 0.5, SIGMA_LATE, 5.0]
OUT = DEFAULT_CONFIG.artifact_dir


def _to_display(img):
    """Take channel 0 (for multichannel images) and clip to [0, 1] for imshow."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    return np.clip(img, 0.0, 1.0)


def visualize_pipeline(image, sample_idx, output_dir=OUT):
    """
    4-panel figure showing one sample through the diffusion preprocessing pipeline.

    Panels:
      1. Raw preprocessing output (loaded .npy, [0,1])
      2. After forward_transform (minmax normalization — visually identical for [0,1] data)
      3. Noisy at sigma=SIGMA_EARLY (structure still clearly visible)
      4. Noisy at sigma=SIGMA_LATE (mostly random noise)

    Args:
        image: (H, W, C) single sample, already sliced from the (N, H, W, C) array.
    """
    stage1 = _to_display(image)
    stage2 = _to_display(forward_transform(image[np.newaxis]).numpy()[0])

    noisy_early, _ = edm_forward_diffuse(image[np.newaxis], SIGMA_EARLY)
    stage3 = _to_display(noisy_early[0])

    noisy_late, _ = edm_forward_diffuse(image[np.newaxis], SIGMA_LATE)
    stage4 = _to_display(noisy_late[0])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (stage1, "1. Preprocessing output\n[0, 1]"),
        (stage2, "2. forward_transform\nminmax [0, 1]"),
        (stage3, f"3. Noisy σ={SIGMA_EARLY}\n(early — structure visible)"),
        (stage4, f"4. Noisy σ={SIGMA_LATE}\n(late — mostly noise)"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.suptitle(f"Diffusion Preprocessing Pipeline — Sample {sample_idx}", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"diffusion_pipeline_sample_{sample_idx}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pipeline figure saved to {out_path}")


def visualize_noise_journey(image, sigmas, output_path):
    """
    Panel grid showing one sample forward-noised at each sigma in `sigmas`
    (ascending), using the EDM2 forward process x_t = x0 + sigma*eps.

    Args:
        image: (H, W, C) single sample.
        sigmas: ascending noise levels; 0.0 renders the clean image unchanged.
        output_path: path to save the figure.
    """
    fig, axes = plt.subplots(1, len(sigmas), figsize=(3 * len(sigmas), 3.2))
    for ax, sigma in zip(axes, sigmas):
        if sigma == 0.0:
            noisy = image[np.newaxis]
        else:
            noisy, _ = edm_forward_diffuse(image[np.newaxis], sigma)
        ax.imshow(_to_display(noisy[0]), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"σ={sigma:g}", fontsize=10)
        ax.axis('off')

    plt.suptitle("EDM2 Forward Noising Journey", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Noise journey figure saved to {output_path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    print("Loading preprocessed data...")
    X = np.load(os.path.join(OUT, "resized_expressions.npy")).astype(np.float32)
    print(f"  Shape: {X.shape}  Range: [{X.min():.4f}, {X.max():.4f}]")

    set_data_range(X.min(), X.max())

    for i in SAMPLE_INDICES:
        print(f"\n--- Sample {i} ---")
        visualize_pipeline(X[i], sample_idx=i)
        visualize_noise_journey(
            X[i],
            NOISE_SIGMAS,
            output_path=os.path.join(OUT, f"diffusion_noise_journey_sample_{i}.png"),
        )

    print("\nDone. Outputs in", OUT)
