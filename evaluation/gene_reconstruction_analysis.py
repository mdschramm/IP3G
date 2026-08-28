"""
Gene expression reconstruction quality analysis — how much information is lost
where pixels collide, for any (width, height, channels) preprocessing config.

Two analyses:

  1. Overflow analysis — splits genes into "exact" (dedicated per-pixel channel
     slot, or alone in the shared/overflow slot) vs "averaged" (2+ genes sharing
     the shared/overflow slot at the same pixel — see
     preprocessing.gene_vector_reconstruction.compute_exact_mask), and compares
     their ANOVA F-statistics to test whether the pipeline's importance-ordered
     assignment pushes the least discriminative genes into the averaged group.
     For a single-channel config (channels=1) this reduces to singleton-pixel
     genes (exact) vs collision-pixel genes (averaged).

  2. Reconstruction fidelity — round-trips the preprocessed images back to gene
     expression vectors via reconstruct_gene_vectors, then computes per-gene Pearson r
     between original and reconstructed expressions.

     Recommended metric: per-gene Pearson r across samples.
       - Exact genes: reconstruction is mathematically exact (pixel_val * channel_scale
         = original_expression), so r ≈ 1.0. Any deviation indicates a float32 precision
         issue in the pipeline.
       - Averaged genes: r < 1.0 quantifies signal loss. The pixel stores the mean of
         all genes sharing that slot; reconstructed[:, g] = that mean for every gene g
         at the pixel. r measures how correlated each gene is with the group average.
       - R² = r² gives "fraction of variance preserved" — the headline information-loss number.

REQUIRES:
    output/preprocessing/data.npy                    — (N_samples, N_genes) RSEM matrix (shared)
    output/preprocessing/gene_f_stats.npy             — (N_genes,) ANOVA F-statistic (shared)
    <config.artifact_dir>/gene_pixel_channel.npy      — (N_genes, 3) [px, py, ch] per gene
    <config.artifact_dir>/channel_scales.npy          — (channels,) per-channel normalization
    <config.artifact_dir>/resized_expressions.npy     — (N_samples, H, W, channels) images

USAGE:
    python -m evaluation.gene_reconstruction_analysis
    python -m evaluation.gene_reconstruction_analysis --width 256 --height 256 --channels 1
    python -m evaluation.gene_reconstruction_analysis --width 512 --height 512 --channels 1
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from preprocessing.gene_vector_reconstruction import (
    reconstruct_gene_vectors,
    load_reconstruction_artifacts,
    compute_exact_mask,
)
from preprocessing.artifact_paths import PreprocessingConfig, DATA_PATH, GENE_F_STATS_PATH


# ---------------------------------------------------------------------------
# Analysis 1 — overflow gene counts and discriminative power
# ---------------------------------------------------------------------------

def analyze_overflow_genes(gene_pixel_channel, gene_f_stats, n_channels, out_path):
    """
    Report how many genes get an exact vs. averaged reconstruction (see
    compute_exact_mask), and compare their ANOVA F-statistics.

    Args:
        gene_pixel_channel: (N_genes, 3) int32 — [px, py, ch] per gene
        gene_f_stats:       (N_genes,) float32 — ANOVA F-statistic per gene
        n_channels:         number of channels the config was built with
        out_path:           path to save figure

    Returns:
        exact_mask: (N_genes,) bool — True for genes with an exact reconstruction
    """
    exact_mask = compute_exact_mask(gene_pixel_channel, n_channels)
    n_total = len(gene_pixel_channel)
    n_exact = int(exact_mask.sum())
    n_lossy = n_total - n_exact

    if n_channels > 1:
        exact_label = f"ch 0-{n_channels - 2} (exact)"
        lossy_label = f"ch {n_channels - 1} (shared overflow)"
    else:
        exact_label = "singleton-pixel (exact)"
        lossy_label = "collision-pixel (averaged)"

    print(f"\n{'='*60}")
    print("RECONSTRUCTION EXACTNESS ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total genes           : {n_total:,}")
    print(f"  Exact — {exact_label:<24}: {n_exact:,}  ({100*n_exact/n_total:.1f}%)")
    print(f"  Averaged — {lossy_label:<21}: {n_lossy:,}  ({100*n_lossy/n_total:.1f}%)")

    if n_channels > 1:
        print(f"\n  Per-channel gene counts:")
        for ch in range(n_channels):
            n = int((gene_pixel_channel[:, 2] == ch).sum())
            tag = " ← overflow (averaged where shared)" if ch == n_channels - 1 else ""
            print(f"    ch {ch:2d} : {n:,}{tag}")

    f_exact = gene_f_stats[exact_mask]
    f_lossy = gene_f_stats[~exact_mask]

    print(f"\n  F-statistic — {exact_label}:")
    print(f"    median={np.median(f_exact):.3f}  mean={np.mean(f_exact):.3f}"
          f"  p90={np.percentile(f_exact, 90):.3f}")
    print(f"  F-statistic — {lossy_label}:")
    print(f"    median={np.median(f_lossy):.3f}  mean={np.mean(f_lossy):.3f}"
          f"  p90={np.percentile(f_lossy, 90):.3f}")

    u, pval = stats.mannwhitneyu(f_exact, f_lossy, alternative='two-sided')
    direction = "higher" if np.median(f_exact) > np.median(f_lossy) else "lower"
    print(f"\n  Mann-Whitney U={u:.0f},  p={pval:.4g}")
    if pval < 0.05:
        print(f"  → Exact-reconstruction genes have significantly {direction} F-statistics (p<0.05).")
    else:
        print(f"  → No significant F-statistic difference between groups (p≥0.05).")

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(["exact", "averaged"], [n_exact, n_lossy], color=["steelblue", "tomato"])
    for i, v in enumerate([n_exact, n_lossy]):
        axes[0].text(i, v, f"{v:,}\n({100*v/n_total:.1f}%)", ha='center', va='bottom', fontsize=9)
    axes[0].set_ylabel("Number of genes")
    axes[0].set_title(f"Genes: exact vs. averaged reconstruction\n{exact_label} / {lossy_label}")

    max_f = min(float(np.percentile(np.concatenate([f_exact, f_lossy]), 99)), 500.0)
    bins  = np.linspace(0, max_f, 60)
    axes[1].hist(f_exact, bins=bins, alpha=0.6, density=True,
                 color='steelblue', label=f'Exact (n={n_exact:,})')
    axes[1].hist(f_lossy, bins=bins, alpha=0.6, density=True,
                 color='tomato',    label=f'Averaged (n={n_lossy:,})')
    axes[1].set_xlabel("ANOVA F-statistic")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"F-stat distribution by reconstruction type\n(Mann-Whitney p={pval:.3g})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")

    return exact_mask


# ---------------------------------------------------------------------------
# Analysis 2 — round-trip reconstruction fidelity
# ---------------------------------------------------------------------------

def reconstruction_fidelity(data, images, gene_pixel_channel, channel_scales, n_channels,
                            out_path, n_samples=2000):
    """
    Compute per-gene Pearson r between original expressions and the round-trip
    reconstruction from the multichannel images.

    For exact genes the reconstruction is exact so r ≈ 1.0 (any deviation is
    float32 rounding). For averaged genes r < 1.0 quantifies information loss.
    R² = r² is the fraction of cross-sample variance preserved.

    Args:
        data:               (N_samples, N_genes) float — original RSEM expressions
        images:             (N_samples, H, W, n_channels) float32 images
        gene_pixel_channel: (N_genes, 3) int32
        channel_scales:     (n_channels,) float32
        n_channels:         number of channels the config was built with
        out_path:           path to save figure
        n_samples:          number of samples to use; None uses all (default 100)

    Returns:
        r_per_gene: (N_genes,) float32 Pearson r per gene
    """
    print(f"\n{'='*60}")
    print("RECONSTRUCTION FIDELITY (per-gene Pearson r)")
    print(f"{'='*60}")

    n_total = len(data)
    if n_samples is not None and n_samples < n_total:
        rng = np.random.default_rng()
        idx    = rng.choice(n_total, size=n_samples, replace=False)
        data   = data[idx]
        images = images[idx]
        print(f"  Sampling {n_samples} / {n_total} samples (random) …")
    else:
        print(f"  Using all {n_total} samples …")

    print("  Reconstructing gene vectors …")

    reconstructed = reconstruct_gene_vectors(images, gene_pixel_channel, channel_scales)
    del images  # free 4.1 GB before the Pearson computation

    # Vectorised Pearson r per gene (column-wise).
    # float32 is sufficient — values are in [0, 1] after normalization.
    # Compute mean and subtract in-place to avoid duplicate allocations.
    d  = data.astype(np.float32);  d  -= d.mean(axis=0)
    rc = reconstructed;            rc -= rc.mean(axis=0)
    del reconstructed

    var_data   = np.einsum('ij,ij->j', d, d)   # proportional to per-gene variance
    var_recon  = np.einsum('ij,ij->j', rc, rc)
    num        = np.einsum('ij,ij->j', d, rc)
    den        = np.sqrt(var_data * var_recon)
    r_per_gene = np.where(den > 1e-10, num / den, np.nan).astype(np.float32)
    del d, rc

    # Genes with near-zero variance across the sampled observations have undefined
    # Pearson r (0/0 → NaN). These are lowly-expressed genes that appear constant in
    # any small sample window — not a reconstruction failure. Exclude them from the
    # fidelity stats but report how many were filtered.
    n_undefined = int(np.isnan(r_per_gene).sum())
    var_mask    = ~np.isnan(r_per_gene)   # True where r is defined
    print(f"\n  Genes with undefined r (near-zero variance): {n_undefined:,}"
          f"  ({100*n_undefined/len(r_per_gene):.1f}%)")

    exact_mask = compute_exact_mask(gene_pixel_channel, n_channels)
    r_exact = r_per_gene[exact_mask & var_mask]
    r_lossy = r_per_gene[~exact_mask & var_mask]

    if n_channels > 1:
        exact_label = f"Channels 0-{n_channels - 2}"
        lossy_label = f"Channel {n_channels - 1} (overflow)"
    else:
        exact_label = "Singleton pixels"
        lossy_label = "Collision pixels (averaged)"

    print(f"\n  {exact_label} — exact reconstruction (n={len(r_exact):,} variable genes):")
    print(f"    mean r={r_exact.mean():.5f}  median r={np.median(r_exact):.5f}")
    print(f"    mean R²={np.mean(r_exact**2):.5f}")
    print(f"    genes with r < 0.99: {int((r_exact < 0.99).sum()):,}  (expected ~0 — float32 noise only)")

    print(f"\n  {lossy_label} — averaged (n={len(r_lossy):,} variable genes):")
    print(f"    mean r={r_lossy.mean():.4f}  median r={np.median(r_lossy):.4f}")
    r2_lossy = r_lossy ** 2
    print(f"    mean R²={r2_lossy.mean():.4f}  (fraction of variance preserved)")
    print(f"    genes with r < 0.50: {int((r_lossy < 0.50).sum()):,}")
    print(f"    genes with r < 0.00: {int((r_lossy < 0.00).sum()):,}")

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bins = np.linspace(-1, 1, 80)
    axes[0].hist(r_exact, bins=bins, alpha=0.7, density=True,
                 color='steelblue', label=f'{exact_label} (n={len(r_exact):,})')
    axes[0].hist(r_lossy, bins=bins, alpha=0.7, density=True,
                 color='tomato',    label=f'{lossy_label} (n={len(r_lossy):,})')
    axes[0].set_xlabel("Pearson r")
    axes[0].set_ylabel("Fraction of genes per bin width")
    axes[0].set_title(f"Round-trip Pearson r (variable genes only)\n"
                      f"{n_undefined:,} near-constant genes excluded")
    axes[0].legend()
    axes[0].axvline(1.0, color='gray', linestyle='--', linewidth=0.8)

    axes[1].hist(r_lossy, bins=60, color='tomato', alpha=0.8, density=True)
    axes[1].set_xlabel("Pearson r")
    axes[1].set_ylabel("Fraction of genes per bin width")
    axes[1].set_title(f"{lossy_label} (variable only)\nmean r={r_lossy.mean():.3f}"
                      f"  mean R²={r2_lossy.mean():.3f}")
    axes[1].axvline(float(np.median(r_lossy)), color='black', linestyle='--',
                    linewidth=1, label=f'median={np.median(r_lossy):.3f}')
    axes[1].legend()

    if n_channels > 1:
        r2_by_ch = []
        for ch in range(n_channels):
            mask = (gene_pixel_channel[:, 2] == ch) & var_mask
            r2_by_ch.append(float(np.mean(r_per_gene[mask] ** 2)) if mask.sum() > 0 else np.nan)
        colors = ['tomato' if ch == n_channels - 1 else 'steelblue' for ch in range(n_channels)]
        axes[2].bar(range(n_channels), r2_by_ch, color=colors)
        axes[2].set_xlabel("Channel")
        axes[2].set_ylabel("Mean R²  (variable genes only)")
        axes[2].set_title("Mean R² per channel\n(fraction of variance preserved)")
        axes[2].set_xticks(range(n_channels))
        axes[2].set_ylim(0, 1.05)
        axes[2].axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    else:
        # No per-channel breakdown to show with a single channel — instead show
        # fidelity as a function of collision multiplicity (how many genes share
        # the pixel), which is the informative axis here.
        unique_px, px_counts = np.unique(gene_pixel_channel[:, :2], axis=0, return_counts=True)
        count_lookup = {(int(p[0]), int(p[1])): c for p, c in zip(unique_px, px_counts)}
        gene_collision_count = np.array(
            [count_lookup[(int(p[0]), int(p[1]))] for p in gene_pixel_channel[:, :2]]
        )
        bin_edges = [1, 2, 3, 4, 5, int(gene_collision_count.max()) + 1]
        labels, r2_by_bin = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (gene_collision_count >= lo) & (gene_collision_count < hi) & var_mask
            if mask.sum() == 0:
                continue
            labels.append(str(lo) if hi == lo + 1 else f"{lo}+")
            r2_by_bin.append(float(np.mean(r_per_gene[mask] ** 2)))
        colors = ['steelblue' if l == '1' else 'tomato' for l in labels]
        axes[2].bar(labels, r2_by_bin, color=colors)
        axes[2].set_xlabel("Genes sharing the pixel")
        axes[2].set_ylabel("Mean R²  (variable genes only)")
        axes[2].set_title("Mean R² by collision multiplicity")
        axes[2].set_ylim(0, 1.05)
        axes[2].axhline(1.0, color='gray', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")

    return r_per_gene


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gene reconstruction fidelity analysis for a given preprocessing config"
    )
    parser.add_argument("--width", type=int, default=PreprocessingConfig().width)
    parser.add_argument("--height", type=int, default=PreprocessingConfig().height)
    parser.add_argument("--channels", type=int, default=PreprocessingConfig().channels)
    args = parser.parse_args()

    config = PreprocessingConfig(width=args.width, height=args.height, channels=args.channels)
    OUT_PREPROCESSING = config.artifact_dir
    OUT_EVALUATION = config.evaluation_dir
    os.makedirs(OUT_EVALUATION, exist_ok=True)

    print(f"Config: {config.tag}\n")

    print("Loading preprocessing artifacts …")
    gpc, channel_scales = load_reconstruction_artifacts(OUT_PREPROCESSING)
    gene_f_stats = np.load(GENE_F_STATS_PATH)
    print(f"  gene_pixel_channel : {gpc.shape}")
    print(f"  channel_scales     : {channel_scales}")
    print(f"  gene_f_stats       : {gene_f_stats.shape}")

    print("\n[1/2] Reconstruction exactness analysis …")
    analyze_overflow_genes(
        gpc, gene_f_stats, config.channels,
        out_path=f"{OUT_EVALUATION}/overflow_analysis.png",
    )

    print("\n[2/2] Reconstruction fidelity …")
    images_path = config.resized_expressions_path

    for path in [DATA_PATH, images_path]:
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found — run "
                  f"`python -m preprocessing.prepare_training_data --width {config.width} "
                  f"--height {config.height} --channels {config.channels}` first.")
            raise SystemExit(1)

    print("  Loading data.npy and resized_expressions.npy …")
    data   = np.load(DATA_PATH)
    images = np.load(images_path, mmap_mode='r')  # memory-mapped — only sampled rows are read
    print(f"  data   : {data.shape}")
    print(f"  images : {images.shape}")

    reconstruction_fidelity(
        data, images, gpc, channel_scales, config.channels,
        out_path=f"{OUT_EVALUATION}/reconstruction_fidelity.png",
    )

    print("\nDone.")
