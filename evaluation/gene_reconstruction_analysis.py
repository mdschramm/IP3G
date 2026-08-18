"""
Gene expression reconstruction quality analysis for the 16-channel image pipeline.

Two analyses:

  1. Overflow analysis — counts genes assigned to channel 15 (averaged) vs channels
     0-14 (exact one-to-one), and compares their ANOVA F-statistics to test whether
     the pipeline's importance-ordered assignment pushes the least discriminative genes
     into the overflow slot.

  2. Reconstruction fidelity — round-trips the preprocessed images back to gene
     expression vectors via reconstruct_gene_vectors, then computes per-gene Pearson r
     between original and reconstructed expressions.

     Recommended metric: per-gene Pearson r across samples.
       - Channels 0-14: reconstruction is mathematically exact (pixel_val * channel_scale
         = original_expression), so r ≈ 1.0. Any deviation indicates a float32 precision
         issue in the pipeline.
       - Channel 15: r < 1.0 quantifies signal loss. The pixel stores the mean of all
         overflow genes at that location; reconstructed[:, g] = that mean for every gene g
         at the pixel. r measures how correlated each gene is with the group average.
       - R² = r² gives "fraction of variance preserved" — the headline information-loss number.

REQUIRES (output/preprocessing/):
    data.npy                 — (N_samples, N_genes) RSEM count matrix
    gene_pixel_channel.npy   — (N_genes, 3) [px, py, ch] per gene
    gene_f_stats.npy         — (N_genes,) ANOVA F-statistic per gene
    channel_scales.npy       — (16,) per-channel normalization factors
    resized_expressions.npy  — (N_samples, 128, 128, 16) multichannel images

USAGE:
    python -m evaluation.gene_reconstruction_analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from preprocessing.gene_vector_reconstruction import (
    reconstruct_gene_vectors,
    load_reconstruction_artifacts,
)

OUT_PREPROCESSING = "output/preprocessing"
OUT_EVALUATION    = "output/evaluation"


# ---------------------------------------------------------------------------
# Analysis 1 — overflow gene counts and discriminative power
# ---------------------------------------------------------------------------

def analyze_overflow_genes(gene_pixel_channel, gene_f_stats, out_path):
    """
    Report how many genes are in channel 15 (overflow / averaged) vs channels 0-14
    (exact), and compare their ANOVA F-statistics.

    Args:
        gene_pixel_channel: (N_genes, 3) int32 — [px, py, ch] per gene
        gene_f_stats:       (N_genes,) float32 — ANOVA F-statistic per gene
        out_path:           path to save figure

    Returns:
        overflow_mask: (N_genes,) bool — True for ch-15 genes
    """
    overflow_mask = gene_pixel_channel[:, 2] == 15
    n_total    = len(gene_pixel_channel)
    n_overflow = int(overflow_mask.sum())
    n_exact    = n_total - n_overflow

    print(f"\n{'='*60}")
    print("OVERFLOW GENE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total genes        : {n_total:,}")
    print(f"  Ch 0-14 (exact)    : {n_exact:,}  ({100*n_exact/n_total:.1f}%)")
    print(f"  Ch 15 (overflow)   : {n_overflow:,}  ({100*n_overflow/n_total:.1f}%)")

    print(f"\n  Per-channel gene counts:")
    for ch in range(16):
        n = int((gene_pixel_channel[:, 2] == ch).sum())
        tag = " ← overflow (averaged)" if ch == 15 else ""
        print(f"    ch {ch:2d} : {n:,}{tag}")

    f_exact    = gene_f_stats[~overflow_mask]
    f_overflow = gene_f_stats[overflow_mask]

    print(f"\n  F-statistic — ch 0-14 (exact):")
    print(f"    median={np.median(f_exact):.3f}  mean={np.mean(f_exact):.3f}"
          f"  p90={np.percentile(f_exact, 90):.3f}")
    print(f"  F-statistic — ch 15 (overflow):")
    print(f"    median={np.median(f_overflow):.3f}  mean={np.mean(f_overflow):.3f}"
          f"  p90={np.percentile(f_overflow, 90):.3f}")

    u, pval = stats.mannwhitneyu(f_exact, f_overflow, alternative='two-sided')
    direction = "higher" if np.median(f_exact) > np.median(f_overflow) else "lower"
    print(f"\n  Mann-Whitney U={u:.0f},  p={pval:.4g}")
    if pval < 0.05:
        print(f"  → Ch 0-14 genes have significantly {direction} F-statistics (p<0.05).")
    else:
        print(f"  → No significant F-statistic difference between groups (p≥0.05).")

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ch_counts = [(gene_pixel_channel[:, 2] == ch).sum() for ch in range(16)]
    colors = ['tomato' if ch == 15 else 'steelblue' for ch in range(16)]
    axes[0].bar(range(16), ch_counts, color=colors)
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Number of genes")
    axes[0].set_title("Genes per channel\n(red = ch 15, averaged overflow)")
    axes[0].set_xticks(range(16))

    max_f = min(float(np.percentile(np.concatenate([f_exact, f_overflow]), 99)), 500.0)
    bins  = np.linspace(0, max_f, 60)
    axes[1].hist(f_exact,    bins=bins, alpha=0.6, density=True,
                 color='steelblue', label=f'Ch 0-14 exact (n={n_exact:,})')
    axes[1].hist(f_overflow, bins=bins, alpha=0.6, density=True,
                 color='tomato',    label=f'Ch 15 overflow (n={n_overflow:,})')
    axes[1].set_xlabel("ANOVA F-statistic")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"F-stat distribution by channel group\n(Mann-Whitney p={pval:.3g})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")

    return overflow_mask


# ---------------------------------------------------------------------------
# Analysis 2 — round-trip reconstruction fidelity
# ---------------------------------------------------------------------------

def reconstruction_fidelity(data, images, gene_pixel_channel, channel_scales, out_path,
                            n_samples=2000):
    """
    Compute per-gene Pearson r between original expressions and the round-trip
    reconstruction from the multichannel images.

    For channels 0-14 the reconstruction is exact so r ≈ 1.0 (any deviation is
    float32 rounding). For channel 15 r < 1.0 quantifies information loss.
    R² = r² is the fraction of cross-sample variance preserved.

    Args:
        data:               (N_samples, N_genes) float — original RSEM expressions
        images:             (N_samples, 128, 128, 16) float32 multichannel images
        gene_pixel_channel: (N_genes, 3) int32
        channel_scales:     (16,) float32
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

    overflow_mask = gene_pixel_channel[:, 2] == 15
    r_exact    = r_per_gene[~overflow_mask & var_mask]
    r_overflow = r_per_gene[ overflow_mask & var_mask]

    print(f"\n  Channels 0-14 — exact reconstruction (n={len(r_exact):,} variable genes):")
    print(f"    mean r={r_exact.mean():.5f}  median r={np.median(r_exact):.5f}")
    print(f"    mean R²={np.mean(r_exact**2):.5f}")
    print(f"    genes with r < 0.99: {int((r_exact < 0.99).sum()):,}  (expected ~0 — float32 noise only)")

    print(f"\n  Channel 15 — overflow / averaged (n={len(r_overflow):,} variable genes):")
    print(f"    mean r={r_overflow.mean():.4f}  median r={np.median(r_overflow):.4f}")
    r2_overflow = r_overflow ** 2
    print(f"    mean R²={r2_overflow.mean():.4f}  (fraction of variance preserved)")
    print(f"    genes with r < 0.50: {int((r_overflow < 0.50).sum()):,}")
    print(f"    genes with r < 0.00: {int((r_overflow < 0.00).sum()):,}")

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bins = np.linspace(-1, 1, 80)
    axes[0].hist(r_exact,    bins=bins, alpha=0.7, density=True,
                 color='steelblue', label=f'Ch 0-14 (n={len(r_exact):,})')
    axes[0].hist(r_overflow, bins=bins, alpha=0.7, density=True,
                 color='tomato',    label=f'Ch 15 (n={len(r_overflow):,})')
    axes[0].set_xlabel("Pearson r")
    axes[0].set_ylabel("Fraction of genes per bin width")
    axes[0].set_title(f"Round-trip Pearson r (variable genes only)\n"
                      f"{n_undefined:,} near-constant genes excluded")
    axes[0].legend()
    axes[0].axvline(1.0, color='gray', linestyle='--', linewidth=0.8)

    axes[1].hist(r_overflow, bins=60, color='tomato', alpha=0.8, density=True)
    axes[1].set_xlabel("Pearson r")
    axes[1].set_ylabel("Fraction of genes per bin width")
    axes[1].set_title(f"Ch 15 overflow genes (variable only)\nmean r={r_overflow.mean():.3f}"
                      f"  mean R²={r2_overflow.mean():.3f}")
    axes[1].axvline(float(np.median(r_overflow)), color='black', linestyle='--',
                    linewidth=1, label=f'median={np.median(r_overflow):.3f}')
    axes[1].legend()

    r2_by_ch = []
    for ch in range(16):
        mask = (gene_pixel_channel[:, 2] == ch) & var_mask
        r2_by_ch.append(float(np.mean(r_per_gene[mask] ** 2)) if mask.sum() > 0 else np.nan)
    colors = ['tomato' if ch == 15 else 'steelblue' for ch in range(16)]
    axes[2].bar(range(16), r2_by_ch, color=colors)
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Mean R²  (variable genes only)")
    axes[2].set_title("Mean R² per channel\n(fraction of variance preserved)")
    axes[2].set_xticks(range(16))
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
    os.makedirs(OUT_EVALUATION, exist_ok=True)

    print("Loading preprocessing artifacts …")
    gpc, channel_scales = load_reconstruction_artifacts(OUT_PREPROCESSING)
    gene_f_stats = np.load(f"{OUT_PREPROCESSING}/gene_f_stats.npy")
    print(f"  gene_pixel_channel : {gpc.shape}")
    print(f"  channel_scales     : {channel_scales}")
    print(f"  gene_f_stats       : {gene_f_stats.shape}")

    print("\n[1/2] Overflow gene analysis …")
    analyze_overflow_genes(
        gpc, gene_f_stats,
        out_path=f"{OUT_EVALUATION}/overflow_analysis.png",
    )

    print("\n[2/2] Reconstruction fidelity …")
    data_path   = f"{OUT_PREPROCESSING}/data.npy"
    images_path = f"{OUT_PREPROCESSING}/resized_expressions.npy"

    for path in [data_path, images_path]:
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found — run prepare_training_data.py first.")
            raise SystemExit(1)

    print("  Loading data.npy and resized_expressions.npy …")
    data   = np.load(data_path)
    images = np.load(images_path, mmap_mode='r')  # memory-mapped — only sampled rows are read
    print(f"  data   : {data.shape}")
    print(f"  images : {images.shape}")

    reconstruction_fidelity(
        data, images, gpc, channel_scales,
        out_path=f"{OUT_EVALUATION}/reconstruction_fidelity.png",
    )

    print("\nDone.")
