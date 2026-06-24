"""Visualize diffusion-generated samples as a class grid.

Loads the .npy files produced by diffusion_edm_sample.generate_dataset_edm
(or the legacy diffusion_sample.generate_dataset) and plots one row per class,
n_per_class samples per row, with a shared colormap so intensity is comparable
across panels.

Usage:
    python -m evaluation.visualize_diffusion_samples \\
        --samples output/diffusion/diagnostic/samples/diffusion_synthetic_expressions_w3.0.npy \\
        --labels  output/diffusion/diagnostic/samples/diffusion_synthetic_labels_w3.0.npy \\
        --out     output/diffusion/diagnostic/samples/preview_w3.0.png \\
        --n-classes 16 --n-per-class 4
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def visualize_diffusion_samples(
    samples_path,
    labels_path,
    output_path,
    n_classes=16,
    n_per_class=4,
    seed=0,
    guidance_scale=None,
):
    rng = np.random.default_rng(seed)

    X = np.load(samples_path)                          # [N, H, W]
    y = np.argmax(np.load(labels_path), axis=1)        # [N]

    print(f"Samples: {X.shape}  range=[{X.min():.3f}, {X.max():.3f}]")
    print(f"Classes present: {len(np.unique(y))}")

    classes = sorted(np.unique(y))[:n_classes]
    n_rows = len(classes)
    n_cols = n_per_class

    vmin, vmax = X.min(), X.max()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), squeeze=False)

    for row, cls in enumerate(classes):
        pool = np.where(y == cls)[0]
        idx = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)

        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(idx):
                ax.imshow(X[idx[col]], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if col == 0:
                ax.text(
                    -0.08, 0.5, f"class {cls}",
                    transform=ax.transAxes,
                    ha="right", va="center", fontsize=8,
                )

    # Parse guidance scale from filename if not provided
    if guidance_scale is None:
        m = re.search(r"_w([\d.]+)", os.path.basename(samples_path))
        guidance_scale = m.group(1) if m else "?"

    fig.suptitle(f"Diffusion samples  (w={guidance_scale})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples",
        default="output/diffusion/diagnostic/samples/diffusion_synthetic_expressions_w3.0.npy",
        help="Path to generated images .npy  [N, H, W]",
    )
    p.add_argument(
        "--labels",
        default="output/diffusion/diagnostic/samples/diffusion_synthetic_labels_w3.0.npy",
        help="Path to one-hot labels .npy  [N, num_classes]",
    )
    p.add_argument(
        "--out",
        default="output/diffusion/diagnostic/samples/preview_w3.0.png",
        help="Output PNG path",
    )
    p.add_argument("--n-classes", type=int, default=16, help="Rows to show (default: 16)")
    p.add_argument("--n-per-class", type=int, default=4, help="Samples per row (default: 4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance-scale", type=str, default=None,
                   help="Guidance scale label for title (inferred from filename if omitted)")
    args = p.parse_args()

    visualize_diffusion_samples(
        samples_path=args.samples,
        labels_path=args.labels,
        output_path=args.out,
        n_classes=args.n_classes,
        n_per_class=args.n_per_class,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
    )
