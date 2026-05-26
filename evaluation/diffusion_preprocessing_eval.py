"""
Diffusion Preprocessing Pipeline Evaluation

Visualizes sample images at each stage of the diffusion pipeline:
  1. Raw preprocessing output [0, 1]
  2. Minmax-normalized (forward_transform) — should look identical for [0,1] data
  3. Forward-noised at t=200 (early noise — structure still visible)
  4. Forward-noised at t=600 (heavy noise — mostly random)

Also generates the full noise journey (t=0..1000) via visualize_diffusion_process.

USAGE:
    python -m evaluation.diffusion_preprocessing_eval
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from diffusion.diffusion_utils import (
    forward_transform,
    set_data_range,
    init_schedule,
    prepare_batch_ordinal,
    visualize_diffusion_process,
)

SAMPLE_INDICES = [0, 100, 500]
NOISE_TIMESTEPS = [0, 200, 400, 600, 800, 1000]
OUT = "output/preprocessing"


def _to_display(img):
    """Squeeze channel dim and clip to [0, 1] for imshow."""
    return np.clip(np.squeeze(img), 0.0, 1.0)


def visualize_pipeline(image, sample_idx, output_dir=OUT):
    """
    4-panel figure showing one sample through the diffusion preprocessing pipeline.

    Panels:
      1. Raw preprocessing output (loaded .npy, [0,1])
      2. After forward_transform (minmax normalization — visually identical for [0,1] data)
      3. Noisy at t=200 (early: structure still clearly visible)
      4. Noisy at t=600 (late: mostly random noise)
    """
    stage1 = _to_display(image)
    stage2 = _to_display(forward_transform(image[np.newaxis, ..., np.newaxis]).numpy()[0])

    noisy_200, _ = prepare_batch_ordinal(image[np.newaxis], np.array([200]))
    stage3 = _to_display(noisy_200[0])

    noisy_600, _ = prepare_batch_ordinal(image[np.newaxis], np.array([600]))
    stage4 = _to_display(noisy_600[0])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (stage1, "1. Preprocessing output\n[0, 1]"),
        (stage2, "2. forward_transform\nminmax [0, 1]"),
        (stage3, "3. Noisy t=200\n(early — structure visible)"),
        (stage4, "4. Noisy t=600\n(late — mostly noise)"),
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


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    print("Loading preprocessed data...")
    X = np.load(os.path.join(OUT, "resized_expressions.npy")).astype(np.float32)
    print(f"  Shape: {X.shape}  Range: [{X.min():.4f}, {X.max():.4f}]")

    set_data_range(X.min(), X.max())
    init_schedule(1000, kind='cosine')

    for i in SAMPLE_INDICES:
        print(f"\n--- Sample {i} ---")
        visualize_pipeline(X[i], sample_idx=i)
        visualize_diffusion_process(
            X[i],
            NOISE_TIMESTEPS,
            output_path=os.path.join(OUT, f"diffusion_noise_journey_sample_{i}.png"),
        )

    print("\nDone. Outputs in", OUT)
