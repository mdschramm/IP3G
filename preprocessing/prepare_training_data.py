#!/usr/bin/env python
"""
Main script to prepare training data from raw gene expression files.

This script orchestrates the full preprocessing pipeline:
1. Load raw gene expression data
2. Apply t-SNE dimensionality reduction
3. Transform gene expressions into 2D images
4. Prepare one-hot encoded labels

OUTPUT FILES:
- output/preprocessing/resized_expressions.npy: (N_samples, 128, 128) - Image representations
- output/preprocessing/y_primary_disease_or_tissue.npy: (N_samples, N_classes) - One-hot labels
- output/preprocessing/y_primary_site.npy: (N_samples, N_classes) - Alternative labels

USAGE:
    python -m preprocessing.prepare_training_data

The script uses caching (load_if_not_exists) so intermediate results are saved.
To force recalculation, delete the cached .npy files in output/preprocessing/

NOTE: t-SNE coordinates are uniformly scaled to fit within 127×127 pixels before
image creation, so pad_data is the only spatial step needed — no interpolation.
Pixel values are exactly [0, 1] in the output.
"""

import numpy as np
from preprocessing.preprocess_data import (
    load_if_not_exists, 
    calculate_data, 
    load_samples, 
    generate_phenotype_mapping, 
    get_phenotypes, 
    get_y_train,
    GTEX_PHENOTYPE
)
from preprocessing.image_preprocessing import (
    get_tsne_data,
    minimum_bounding_rectangle,
    rotate,
    compute_rotation,
    initialize_image_data,
    create_expression_images_from_tsne,
    pad_data,
    TARGET_SIZE
)

# Visualization functions (save to PNG files)
from preprocessing.visualization import plot_tsne, plot_bounding_box, plot_convex_hull, render_image
from scipy.spatial import ConvexHull


if __name__ == "__main__":
    import os
    OUT = "output/preprocessing"
    os.makedirs(OUT, exist_ok=True)

    print("=" * 80)
    print("GENE EXPRESSION TO IMAGE PREPROCESSING PIPELINE")
    print("=" * 80)

    print("\n[1/8] Loading gene expression data...")
    sample_gene_expressions = load_if_not_exists(f"{OUT}/data.npy", calculate_data)
    print(f"Gene expression data shape: {sample_gene_expressions.shape}")

    print("\n[2/8] Applying t-SNE dimensionality reduction...")
    tsne_results = load_if_not_exists(f"{OUT}/tsne_results.npy",
        get_tsne_data,
        data=sample_gene_expressions)
    print(f"t-SNE results shape: {tsne_results.shape}")

    plot_tsne(tsne_results, output_path=f"{OUT}/tsne_scatter.png")

    print("\n[3/8] Computing minimum bounding rectangle...")
    bbox = minimum_bounding_rectangle(tsne_results)

    plot_bounding_box(tsne_results, bbox, output_path=f"{OUT}/tsne_bounding_box.png")
    hull = ConvexHull(tsne_results)
    plot_convex_hull(tsne_results, hull, output_path=f"{OUT}/tsne_convex_hull.png")

    print("\n[4/8] Rotating, normalizing, and scaling t-SNE coordinates...")
    theta = compute_rotation(bbox)
    rotated_tsne = rotate(tsne_results, origin=bbox[0], theta=theta)
    normalized_tsne = rotated_tsne - np.min(rotated_tsne, axis=0)
    print(f"Normalized t-SNE range: x=[0, {np.max(normalized_tsne[:,0]):.1f}], y=[0, {np.max(normalized_tsne[:,1]):.1f}]")

    # Scale coordinates uniformly to fit within TARGET_SIZE-1 pixels.
    # This ensures create_expression_images_from_tsne produces images ≤ TARGET_SIZE×TARGET_SIZE,
    # so pad_data brings them to exactly TARGET_SIZE×TARGET_SIZE with zero-padding.
    # No interpolation is needed, preserving the exact [0, 1] pixel range.
    scale = (TARGET_SIZE - 1) / np.max(normalized_tsne)
    normalized_tsne = normalized_tsne * scale
    print(f"  Scaled by {scale:.4f} → coordinates fit within {TARGET_SIZE-1}×{TARGET_SIZE-1} px")

    rotated_bbox = rotate(bbox, origin=bbox[0], theta=theta)
    plot_bounding_box(rotated_tsne, rotated_bbox, output_path=f"{OUT}/tsne_rotated_bbox.png")

    print("\n[5/8] Creating expression images from t-SNE coordinates...")
    data, w, h = initialize_image_data(sample_gene_expressions, normalized_tsne)
    print(f"Initial image dimensions: {w+1} x {h+1}")

    data = load_if_not_exists(f"{OUT}/unpadded_expressions.npy",
        create_expression_images_from_tsne,
        sample_gene_expressions=sample_gene_expressions,
        normalized_tsne=normalized_tsne,
        data=data,
        w=w,
        h=h)
    print(f"Expression images shape: {data.shape}")

    print(f"\n[6/8] Padding images to {TARGET_SIZE}×{TARGET_SIZE} (final spatial step — no interpolation)...")
    data = pad_data(data, TARGET_SIZE)

    print(f"\n[7/8] Saving padded images as resized_expressions.npy...")
    out_path = f"{OUT}/resized_expressions.npy"
    if not os.path.exists(out_path):
        np.save(out_path, data)
        print(f"  Saved: {out_path}")
    else:
        data = np.load(out_path)
        print(f"  Loaded from cache: {out_path}")
    print(f"  Final image shape: {data.shape}")
    print(f"  Pixel value range: [{np.min(data):.4f}, {np.max(data):.4f}]  (exact [0,1] — no interpolation)")

    render_image(data[0], output_path=f"{OUT}/sample_image_0.png", title="Sample Expression Image 0")
    render_image(data[100], output_path=f"{OUT}/sample_image_100.png", title="Sample Expression Image 100")
    render_image(data[500], output_path=f"{OUT}/sample_image_500.png", title="Sample Expression Image 500")

    print("\n[8/8] Preparing phenotype labels...")
    samples = load_if_not_exists(f"{OUT}/samples.npy", load_samples)

    phenotype_mapping = load_if_not_exists(f"{OUT}/sample_to_body_site_mapping.json",
        generate_phenotype_mapping)
    sample_body_site_phenotypes = load_if_not_exists(f"{OUT}/sample_body_site_phenotypes.npy",
        get_phenotypes,
        samples=samples,
        sample_to_phenotype=phenotype_mapping)
    y_train_primary_disease_or_tissue = load_if_not_exists(f"{OUT}/y_primary_disease_or_tissue.npy",
        get_y_train,
        phenotypes=sample_body_site_phenotypes)
    print(f"Primary disease/tissue labels shape: {y_train_primary_disease_or_tissue.shape}")
    print(f"Number of classes: {y_train_primary_disease_or_tissue.shape[1]}")

    primary_site_mapping = load_if_not_exists(f"{OUT}/primary_site_mapping.json",
        generate_phenotype_mapping,
        source_file=GTEX_PHENOTYPE,
        target_column=2)
    sample_primary_site_phenotypes = load_if_not_exists(f"{OUT}/sample_primary_site_phenotypes.npy",
        get_phenotypes,
        samples=samples,
        sample_to_phenotype=primary_site_mapping)
    y_train_primary_site = load_if_not_exists(f"{OUT}/y_primary_site.npy",
        get_y_train,
        phenotypes=sample_primary_site_phenotypes)
    print(f"Primary site labels shape: {y_train_primary_site.shape}")
    print(f"Number of classes: {y_train_primary_site.shape[1]}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTraining data prepared:")
    print(f"  - Images: {data.shape} (samples, height, width)")
    print(f"  - Labels (disease/tissue): {y_train_primary_disease_or_tissue.shape}")
    print(f"  - Labels (site): {y_train_primary_site.shape}")
    print(f"\nOutput files saved in {OUT}/")
    print(f"  resized_expressions.npy, y_primary_disease_or_tissue.npy, y_primary_site.npy")
    print("\nReady for model training.")
