"""
Pixel collision analysis for the t-SNE gene-to-pixel mapping.

Picks up from a cached tsne_results.npy and runs the remaining coordinate
pipeline steps (bounding box → rotate → normalize → scale → integer cast),
then reports:

  1. Gene counts per pixel  (how many genes share each pixel)
  2. Summary statistics on those counts
  3. Histogram of the count distribution
  4. Per-pixel ANOVA F-statistic across tissue classes, stratified by
     whether the pixel is a singleton (1 gene) or a collision (>1 genes)

USAGE:
    python -m preprocessing.pixel_collision_analysis

REQUIRES (all in output/preprocessing/):
    tsne_results.npy
    resized_expressions.npy
    sample_body_site_phenotypes.npy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import stats

from preprocessing.image_preprocessing import (
    minimum_bounding_rectangle,
    rotate,
    compute_rotation,
    TARGET_SIZE,
)

OUT = "output/preprocessing"


# ---------------------------------------------------------------------------
# Step 1 – reproduce the coordinate pipeline from prepare_training_data.py
# ---------------------------------------------------------------------------

def derive_integer_pixel_coords(tsne_results: np.ndarray) -> np.ndarray:
    """
    Run bounding-box → rotate → normalize → scale → floor on tsne_results.

    Mirrors lines 74-93 of prepare_training_data.py exactly so that the
    integer pixel coordinates here match what create_expression_images_from_tsne
    used when building resized_expressions.npy.

    Args:
        tsne_results: Raw t-SNE output, shape (N_genes, 2)

    Returns:
        pixel_coords: Integer pixel coordinates, shape (N_genes, 2)
            Each row is (px, py) — the pixel a gene maps to.
    """
    bbox = minimum_bounding_rectangle(tsne_results)
    theta = compute_rotation(bbox)
    rotated = rotate(tsne_results, origin=bbox[0], theta=theta)
    normalized = rotated - np.min(rotated, axis=0)

    scale = (TARGET_SIZE - 1) / np.max(normalized)
    normalized = normalized * scale

    # create_expression_images_from_tsne casts with int(), which truncates
    pixel_coords = normalized.astype(int)
    return pixel_coords


# ---------------------------------------------------------------------------
# Step 2 – count genes per pixel
# ---------------------------------------------------------------------------

def count_genes_per_pixel(pixel_coords: np.ndarray):
    """
    For each unique (px, py) pair, count how many genes map there.

    Args:
        pixel_coords: shape (N_genes, 2)

    Returns:
        unique_pixels: shape (M, 2)  — the M distinct pixel locations
        counts:        shape (M,)    — number of genes at each location
    """
    unique_pixels, counts = np.unique(pixel_coords, axis=0, return_counts=True)
    return unique_pixels, counts


# ---------------------------------------------------------------------------
# Step 3 – summary statistics and histogram
# ---------------------------------------------------------------------------

def print_count_statistics(counts: np.ndarray) -> None:
    n_pixels = len(counts)
    n_genes  = counts.sum()
    n_collisions = np.sum(counts > 1)
    genes_in_collisions = counts[counts > 1].sum()

    print(f"\n{'='*60}")
    print("GENE-TO-PIXEL COLLISION STATISTICS")
    print(f"{'='*60}")
    print(f"Total genes mapped   : {n_genes:,}")
    print(f"Unique pixels used   : {n_pixels:,}  (of {TARGET_SIZE**2:,} available)")
    print(f"Collision pixels (>1): {n_collisions:,}  ({100*n_collisions/n_pixels:.1f}% of used pixels)")
    print(f"Genes in collisions  : {genes_in_collisions:,}  ({100*genes_in_collisions/n_genes:.1f}% of all genes)")

    print(f"\n--- Count distribution ---")
    print(f"  Min    : {counts.min()}")
    print(f"  Max    : {counts.max()}")
    print(f"  Mean   : {counts.mean():.3f}")
    print(f"  Median : {np.median(counts):.1f}")
    print(f"  Std    : {counts.std():.3f}")

    desc = stats.describe(counts)
    print(f"  Skew   : {desc.skewness:.3f}")
    print(f"  Kurt   : {desc.kurtosis:.3f}")

    print(f"\n--- Percentiles ---")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p:3d}  : {np.percentile(counts, p):.0f}")

    print(f"\n--- Count frequency table (top 10 values) ---")
    vals, freqs = np.unique(counts, return_counts=True)
    for v, f in sorted(zip(vals, freqs), key=lambda x: -x[1])[:10]:
        bar = "#" * min(40, int(40 * f / freqs.max()))
        print(f"  {v:3d} gene(s): {f:5d} pixels  {bar}")


def plot_count_histogram(counts: np.ndarray, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(counts, bins=range(1, counts.max() + 2), edgecolor="black", align="left")
    axes[0].set_xlabel("Genes per pixel")
    axes[0].set_ylabel("Number of pixels")
    axes[0].set_title("Distribution of genes per pixel")
    axes[0].set_yscale("log")

    # Zoomed: only collision pixels
    collision_counts = counts[counts > 1]
    if len(collision_counts):
        axes[1].hist(collision_counts, bins=range(2, collision_counts.max() + 2),
                     edgecolor="black", align="left", color="tomato")
        axes[1].set_xlabel("Genes per pixel")
        axes[1].set_ylabel("Number of pixels")
        axes[1].set_title("Collision pixels only (>1 gene)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved histogram → {out_path}")


# ---------------------------------------------------------------------------
# Step 4 – discriminative power analysis
# ---------------------------------------------------------------------------

def collision_discrimination_analysis(
    pixel_coords: np.ndarray,
    expressions: np.ndarray,
    phenotypes: np.ndarray,
    out_path: str,
) -> None:
    """
    For every active pixel, compute a one-way ANOVA F-statistic across tissue
    classes using the pixel's expression values in resized_expressions.npy.
    Then compare the F distribution for collision vs singleton pixels.

    Args:
        pixel_coords:  (N_genes, 2)  integer gene→pixel map
        expressions:   (N_samples, 128, 128)  from resized_expressions.npy
        phenotypes:    (N_samples,)  tissue class label per sample
        out_path:      path to save the output figure
    """
    # Build the collision mask on the full 128×128 grid
    unique_pixels, counts = count_genes_per_pixel(pixel_coords)

    classes = np.unique(phenotypes)
    n_classes = len(classes)
    class_idx = {c: i for i, c in enumerate(classes)}
    labels = np.array([class_idx[p] for p in phenotypes])

    print(f"\n{'='*60}")
    print("CLASS DISCRIMINATION ANALYSIS")
    print(f"{'='*60}")
    print(f"  Classes     : {n_classes}")
    print(f"  Samples     : {len(phenotypes)}")
    print(f"  Min samples per class: {min(np.sum(labels == i) for i in range(n_classes))}")
    print(f"  Computing ANOVA F-statistic for {len(unique_pixels):,} active pixels …")

    f_stats     = np.full((TARGET_SIZE, TARGET_SIZE), np.nan)
    collision_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)

    for (px, py), cnt in zip(unique_pixels, counts):
        if px >= TARGET_SIZE or py >= TARGET_SIZE:
            continue

        pixel_vals = expressions[:, px, py]   # (N_samples,)
        groups = [pixel_vals[labels == i] for i in range(n_classes)]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            continue

        f, p = stats.f_oneway(*groups)
        f_stats[px, py] = f

        if cnt > 1:
            collision_mask[px, py] = True

    singleton_mask = (~collision_mask) & ~np.isnan(f_stats)

    f_collision = f_stats[collision_mask & ~np.isnan(f_stats)]
    f_singleton = f_stats[singleton_mask]

    print(f"\n  Collision pixels with valid F: {len(f_collision):,}")
    print(f"  Singleton pixels with valid F: {len(f_singleton):,}")

    if len(f_collision) and len(f_singleton):
        stat, pval = stats.mannwhitneyu(f_collision, f_singleton, alternative="two-sided")
        print(f"\n  Median F — collision : {np.median(f_collision):.3f}")
        print(f"  Median F — singleton : {np.median(f_singleton):.3f}")
        print(f"\n  Mann-Whitney U = {stat:.1f},  p = {pval:.4g}")
        if pval < 0.05:
            direction = "higher" if np.median(f_collision) > np.median(f_singleton) else "lower"
            print(f"  → Collision pixels have significantly {direction} discriminative F-statistics (p<0.05).")
        else:
            print("  → No significant difference in discriminative power (p≥0.05).")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # F-statistic heatmap
    im = axes[0].imshow(np.nan_to_num(f_stats), cmap="hot", origin="upper")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("ANOVA F-statistic per pixel")

    # Overlay collision pixels
    overlay = np.zeros((*f_stats.shape, 4))
    overlay[collision_mask] = [0, 0.5, 1, 0.7]   # blue = collision
    axes[1].imshow(np.nan_to_num(f_stats), cmap="hot", origin="upper")
    axes[1].imshow(overlay, origin="upper")
    axes[1].set_title("Collision pixels (blue) on F-map")

    # Overlapping histograms of F-stats
    bins = np.linspace(0, min(np.nanpercentile(f_stats, 99), 200), 60)
    axes[2].hist(f_singleton, bins=bins, alpha=0.6, label=f"Singleton (n={len(f_singleton):,})",
                 density=True, color="steelblue")
    if len(f_collision):
        axes[2].hist(f_collision, bins=bins, alpha=0.6,
                     label=f"Collision (n={len(f_collision):,})",
                     density=True, color="tomato")
    axes[2].set_xlabel("ANOVA F-statistic")
    axes[2].set_ylabel("Density")
    axes[2].set_title("F-stat distribution by pixel type")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved discrimination figure → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[1/4] Loading cached t-SNE results …")
    tsne_results = np.load(f"{OUT}/tsne_results.npy")
    print(f"  tsne_results shape: {tsne_results.shape}  (N_genes × 2)")

    print("\n[2/4] Deriving integer pixel coordinates …")
    pixel_coords = derive_integer_pixel_coords(tsne_results)
    print(f"  Pixel coordinate range: x=[{pixel_coords[:,0].min()}, {pixel_coords[:,0].max()}], "
          f"y=[{pixel_coords[:,1].min()}, {pixel_coords[:,1].max()}]")

    unique_pixels, counts = count_genes_per_pixel(pixel_coords)
    print_count_statistics(counts)

    print("\n[3/4] Saving collision histogram …")
    plot_count_histogram(counts, out_path=f"{OUT}/pixel_collision_histogram.png")

    print("\n[4/4] Loading expression images and phenotypes for discrimination analysis …")
    expr_path       = f"{OUT}/resized_expressions.npy"
    phenotype_path  = f"{OUT}/sample_body_site_phenotypes.npy"

    if not os.path.exists(expr_path):
        print(f"  SKIP: {expr_path} not found — run prepare_training_data.py first.")
    elif not os.path.exists(phenotype_path):
        print(f"  SKIP: {phenotype_path} not found — run prepare_training_data.py first.")
    else:
        expressions = np.load(expr_path)
        phenotypes  = np.load(phenotype_path, allow_pickle=True)
        collision_discrimination_analysis(
            pixel_coords=pixel_coords,
            expressions=expressions,
            phenotypes=phenotypes,
            out_path=f"{OUT}/pixel_collision_discrimination.png",
        )

    print("\nDone.")
