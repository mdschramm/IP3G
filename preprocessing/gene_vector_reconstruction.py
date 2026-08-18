"""
Reverse mapping from 16-channel expression images back to gene expression vectors.

Forward normalization chain (see image_preprocessing.create_multichannel_expression_images_from_tsne):
  1. Genes are assigned to pixel-channel slots in descending F-statistic order.
     Channels 0-14 at each pixel hold exactly one gene; channel 15 holds the
     arithmetic mean of all overflow genes (>14th gene at that pixel).
  2. Per-channel normalization: image[i, px, py, k] = raw_expression / channel_scales[k]

Inverse:
  raw_expression = image[i, px, py, k] * channel_scales[k]

Exact for channels 0-14 (one-to-one gene-pixel-channel mapping).
For channel 15: all overflow genes at a pixel receive the same reconstructed value
(the pixel average), since individual proportions are lost during the forward mean.
This affects ~1% of genes (p99 = 16 genes/pixel from t-SNE collision analysis).

USAGE:
    from preprocessing.gene_vector_reconstruction import (
        reconstruct_gene_vectors, load_reconstruction_artifacts
    )

    gpc, channel_scales = load_reconstruction_artifacts()
    gene_vectors = reconstruct_gene_vectors(synthetic_images, gpc, channel_scales)
    # gene_vectors: (N_samples, N_genes) float32 in RSEM count units
"""

import numpy as np


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


def load_reconstruction_artifacts(preprocessing_dir="output/preprocessing"):
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

    preprocessing_dir = "output/preprocessing"

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
