"""Visualize GAN-generated samples alongside real images for comparison.

The GAN is InfoGAN-style and its latent classes are unsupervised — they don't
correspond directly to real phenotypes. This script supports two modes:

- Default: shows the first N latent classes next to N random real phenotypes.
- With ``--mapping``: reads ``latent_to_phenotype_mapping.csv`` (produced by
  ``python evaluation.py --mode map``) and aligns each latent class to its
  most-likely real phenotype.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_mapping(csv_path):
    """Read latent→phenotype CSV produced by evaluation.py --mode map."""
    mapping = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["latent_class"])] = {
                "phenotype": int(row["mapped_phenotype"]),
                "confidence": float(row["confidence"]),
                "n_samples": int(row["n_samples"]),
            }
    return mapping


def make_comparison_grid(
    real_path,
    real_label_path,
    syn_path,
    syn_label_path,
    output_path,
    n_classes=10,
    n_per_class=4,
    seed=0,
    mapping_csv=None,
):
    rng = np.random.default_rng(seed)

    X_real = np.load(real_path)
    y_real = np.argmax(np.load(real_label_path), axis=1)
    X_syn = np.load(syn_path)
    y_syn = np.argmax(np.load(syn_label_path), axis=1)

    print(f"Real:      {X_real.shape}  range=[{X_real.min():.3f}, {X_real.max():.3f}]  classes={y_real.max() + 1}")
    print(f"Synthetic: {X_syn.shape}  range=[{X_syn.min():.3f}, {X_syn.max():.3f}]  classes={y_syn.max() + 1}")

    available_latents = np.unique(y_syn)

    if mapping_csv and os.path.exists(mapping_csv):
        mapping = load_mapping(mapping_csv)
        # Sort latent classes by mapping confidence (highest first) so the most
        # interpretable rows show up at the top.
        ordered = sorted(
            (lc for lc in available_latents if lc in mapping),
            key=lambda lc: mapping[lc]["confidence"],
            reverse=True,
        )
        latent_classes = np.array(ordered[:n_classes])
        real_classes = np.array([mapping[int(lc)]["phenotype"] for lc in latent_classes])
        confidences = np.array([mapping[int(lc)]["confidence"] for lc in latent_classes])
        title_suffix = "(latent → mapped phenotype, sorted by confidence)"
    else:
        latent_classes = available_latents[: min(n_classes, len(available_latents))]
        real_classes = rng.choice(np.unique(y_real), size=len(latent_classes), replace=False)
        confidences = np.full(len(latent_classes), np.nan)
        title_suffix = "(arbitrary real classes — no mapping provided)"

    n_rows = len(latent_classes)
    n_cols = 2 * n_per_class + 1  # real | spacer | synthetic

    # Use shared colormap range so visual intensity is comparable across panels.
    vmin = min(X_real.min(), X_syn.min())
    vmax = max(X_real.max(), X_syn.max())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (rc, lc) in enumerate(zip(real_classes, latent_classes)):
        real_pool = np.where(y_real == rc)[0]
        syn_pool = np.where(y_syn == lc)[0]

        if len(real_pool) == 0 or len(syn_pool) == 0:
            for c in range(n_cols):
                axes[row, c].axis("off")
            continue

        real_idx = rng.choice(real_pool, size=min(n_per_class, len(real_pool)), replace=False)
        syn_idx = rng.choice(syn_pool, size=min(n_per_class, len(syn_pool)), replace=False)

        for c in range(n_per_class):
            ax = axes[row, c]
            if c < len(real_idx):
                ax.imshow(X_real[real_idx[c]], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if c == 0:
                ax.text(
                    -0.15,
                    0.5,
                    f"phenotype {rc}",
                    transform=ax.transAxes,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                )

        axes[row, n_per_class].axis("off")  # spacer

        for c in range(n_per_class):
            ax = axes[row, n_per_class + 1 + c]
            if c < len(syn_idx):
                ax.imshow(X_syn[syn_idx[c]], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if c == 0:
                conf = confidences[row]
                label = f"latent {int(lc)}"
                if not np.isnan(conf):
                    label += f"  (conf {conf:.2f})"
                ax.text(
                    1.0,
                    1.05,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                )

    axes[0, n_per_class // 2].set_title("REAL", fontsize=12, fontweight="bold")
    axes[0, n_per_class + 1 + n_per_class // 2].set_title("GAN-SYNTHETIC", fontsize=12, fontweight="bold")
    fig.suptitle(f"Real vs. GAN samples {title_suffix}", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", default="output/preprocessing/resized_expressions.npy")
    p.add_argument("--real-labels", default="output/preprocessing/y_primary_disease_or_tissue.npy")
    p.add_argument("--syn", default="output/gan/remote/synthetic_resized_expressions.npy")
    p.add_argument("--syn-labels", default="output/gan/remote/synthetic_latent_classes.npy")
    p.add_argument("--out", default="output/gan/remote/real_vs_gan_comparison.png")
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument("--n-per-class", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mapping",
        default=None,
        help="Optional path to latent_to_phenotype_mapping.csv "
        "(produced by `python evaluation.py --mode map`). "
        "If provided, rows are aligned latent→phenotype and sorted by confidence.",
    )
    args = p.parse_args()

    make_comparison_grid(
        real_path=args.real,
        real_label_path=args.real_labels,
        syn_path=args.syn,
        syn_label_path=args.syn_labels,
        output_path=args.out,
        n_classes=args.n_classes,
        n_per_class=args.n_per_class,
        seed=args.seed,
        mapping_csv=args.mapping,
    )
