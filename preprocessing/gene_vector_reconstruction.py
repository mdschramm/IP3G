"""
Reverse mapping from multichannel expression images back to gene expression vectors.

Works for any channel count, including the single-channel (channels=1) configs
used for the 256x256x1 / 512x512x1 resolution comparison, where it reduces to
plain per-pixel collision averaging.

Forward normalization chain (see image_preprocessing.create_multichannel_expression_images_from_tsne):
  1. Genes are assigned to pixel-channel slots in descending F-statistic order.
     Channels 0..n_channels-2 at each pixel hold exactly one gene; channel
     n_channels-1 holds the arithmetic mean of all overflow genes (any gene beyond
     the n_channels-1'th at that pixel — every gene, when n_channels == 1).
  2. Per-channel normalization: image[i, px, py, k] = raw_expression / channel_scales[k]

Inverse:
  raw_expression = image[i, px, py, k] * channel_scales[k]

Exact for channels 0..n_channels-2 (one-to-one gene-pixel-channel mapping), and for
any gene alone in the overflow channel at its pixel. Lossy only where 2+ genes share
the overflow channel at the same pixel — see compute_exact_mask() below.

USAGE:
    from preprocessing.gene_vector_reconstruction import (
        reconstruct_gene_vectors, load_reconstruction_artifacts
    )

    gpc, channel_scales = load_reconstruction_artifacts()
    gene_vectors = reconstruct_gene_vectors(synthetic_images, gpc, channel_scales)
    # gene_vectors: (N_samples, N_genes) float32 in RSEM count units
"""

import numpy as np

from preprocessing.artifact_paths import DEFAULT_CONFIG


def reconstruct_gene_vectors(images, gene_pixel_channel, channel_scales):
    """
    Reverse map 16-channel expression images → gene expression vectors.

    Args:
        images:             (N, 128, 128, 16) float32 synthetic images in [0, 1]
        gene_pixel_channel: (N_genes, 3) int32 — [px, py, ch] per gene in original
                            gene ordering. Produced by create_multichannel_expression_images_from_tsne.
        channel_scales:     (16,) float32 — per-channel max values from forward normalization.
                            channel_scales[k] = max(data[:,:,:,k]) before division.
                            Set to 1.0 for any all-zero channel.

    Returns:
        gene_vectors: (N, N_genes) float32 in original RSEM count units.
                      For ch 0-14: exact inversion of the forward normalization.
                      For ch 15:   reconstructed value is the pixel average shared by
                                   all overflow genes at that pixel (flat distribution).
    """
    px = gene_pixel_channel[:, 0]         # (N_genes,)
    py = gene_pixel_channel[:, 1]         # (N_genes,)
    ch = gene_pixel_channel[:, 2]         # (N_genes,)

    # Vectorised fancy index: images[:, px, py, ch] → (N, N_genes)
    pixel_vals   = images[:, px, py, ch]              # (N, N_genes)
    scales       = channel_scales[ch][None, :]        # (1, N_genes) broadcasts over N
    gene_vectors = pixel_vals * scales                # (N, N_genes)
    return gene_vectors


def compute_exact_mask(gene_pixel_channel, n_channels):
    """
    Which genes get an exact (lossless) round-trip reconstruction vs. an averaged
    (lossy) one, for any channel count.

    A gene is exact if it holds a dedicated per-pixel slot (channels 0..n_channels-2),
    or if it's the only gene sharing the overflow slot (channel n_channels-1) at its
    pixel — create_multichannel_expression_images_from_tsne only divides by the
    overflow count when that count is > 1, so a lone overflow occupant is untouched.
    A gene is lossy only when 2+ genes share the overflow slot at the same pixel —
    they're then indistinguishable in the reconstructed value (the pixel average).

    For n_channels == 1 every gene is nominally "in the overflow slot" (there's only
    one channel), so this reduces to: exact == the gene's pixel has no collision,
    lossy == it does — i.e. the same singleton/collision split as
    evaluation.pixel_collision_analysis.count_genes_per_pixel.

    Args:
        gene_pixel_channel: (N_genes, 3) int32 — [px, py, ch] per gene
        n_channels: number of channels the config was built with

    Returns:
        exact_mask: (N_genes,) bool — True where reconstruction is exact
    """
    ch = gene_pixel_channel[:, 2]
    overflow_ch = n_channels - 1
    in_overflow = ch == overflow_ch

    exact_mask = np.ones(len(gene_pixel_channel), dtype=bool)
    if np.any(in_overflow):
        pxpy = gene_pixel_channel[in_overflow][:, :2]
        _, inverse, counts = np.unique(pxpy, axis=0, return_inverse=True, return_counts=True)
        shared = counts[inverse] > 1
        exact_mask[in_overflow] = ~shared
    return exact_mask


def load_reconstruction_artifacts(preprocessing_dir=DEFAULT_CONFIG.artifact_dir):
    """Load the artifacts needed for reverse mapping.

    Args:
        preprocessing_dir: path to the preprocessing output directory

    Returns:
        gene_pixel_channel: (N_genes, 3) int32 — [px, py, ch] per gene
        channel_scales:     (16,) float32 — per-channel normalization factors
    """
    gpc = np.load(f"{preprocessing_dir}/gene_pixel_channel.npy")   # (N_genes, 3)
    cs  = np.load(f"{preprocessing_dir}/channel_scales.npy")        # (16,)
    return gpc, cs


if __name__ == "__main__":
    import os

    preprocessing_dir = DEFAULT_CONFIG.artifact_dir

    resized_path = os.path.join(preprocessing_dir, "resized_expressions.npy")
    if not os.path.exists(resized_path):
        print(f"SKIP: {resized_path} not found. Run prepare_training_data.py first.")
    else:
        print("Round-trip test: real images → gene vectors → check range and shape")
        images = np.load(resized_path, mmap_mode='r')[:10].astype(np.float32)
        gpc, channel_scales = load_reconstruction_artifacts(preprocessing_dir)
        gene_vectors = reconstruct_gene_vectors(images, gpc, channel_scales)
        print(f"  images shape:       {images.shape}")
        print(f"  gene_vectors shape: {gene_vectors.shape}")
        print(f"  gene_vectors range: [{gene_vectors.min():.4f}, {gene_vectors.max():.4f}]")
        print(f"  channel_scales[:8]: {channel_scales[:8]}")
        print(f"  Non-zero gene values: {np.sum(gene_vectors > 0):,} / {gene_vectors.size:,}")
        print("Done.")
