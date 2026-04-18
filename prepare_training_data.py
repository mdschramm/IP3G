#!/usr/bin/env python
"""
Main script to prepare training data from raw gene expression files.

This script orchestrates the full preprocessing pipeline:
1. Load raw gene expression data
2. Apply t-SNE dimensionality reduction
3. Transform gene expressions into 2D images
4. Prepare one-hot encoded labels

OUTPUT FILES:
- loaded_data/resized_expressions.npy: (N_samples, 128, 128) - Image representations
- loaded_data/y_primary_disease_or_tissue.npy: (N_samples, N_classes) - One-hot labels
- loaded_data/y_primary_site.npy: (N_samples, N_classes) - Alternative labels

USAGE:
    python prepare_training_data.py

The script uses caching (load_if_not_exists) so intermediate results are saved.
To force recalculation, delete the cached .npy files in loaded_data/
"""

import numpy as np
from preprocess_data import (
    load_if_not_exists, 
    calculate_data, 
    load_samples, 
    generate_phenotype_mapping, 
    get_phenotypes, 
    get_y_train,
    GTEX_PHENOTYPE
)
from image_preprocessing import (
    get_tsne_data, 
    minimum_bounding_rectangle, 
    rotate, 
    compute_rotation, 
    initialize_image_data, 
    create_expression_images_from_tsne, 
    pad_data, 
    resize_images,
    TARGET_SIZE
)

# Visualization functions (save to PNG files)
from visualization import plot_tsne, plot_bounding_box, plot_convex_hull, render_image
from scipy.spatial import ConvexHull


if __name__ == "__main__":
    print("=" * 80)
    print("GENE EXPRESSION TO IMAGE PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load gene expression data
    print("\n[1/8] Loading gene expression data...")
    sample_gene_expressions = load_if_not_exists("loaded_data/data.npy", calculate_data)
    print(f"Gene expression data shape: {sample_gene_expressions.shape}")
    
    # Step 2: Apply t-SNE dimensionality reduction
    print("\n[2/8] Applying t-SNE dimensionality reduction...")
    tsne_results = load_if_not_exists("loaded_data/tsne_results.npy", 
        get_tsne_data, 
        data=sample_gene_expressions)
    print(f"t-SNE results shape: {tsne_results.shape}")
    
    # Visualize t-SNE results
    plot_tsne(tsne_results)
    
    # Step 3: Find minimum bounding rectangle
    print("\n[3/8] Computing minimum bounding rectangle...")
    bbox = minimum_bounding_rectangle(tsne_results)
    
    # Visualize bounding box and convex hull
    plot_bounding_box(tsne_results, bbox)
    hull = ConvexHull(tsne_results)
    plot_convex_hull(tsne_results, hull)
    
    # Step 4: Rotate and normalize t-SNE coordinates
    print("\n[4/8] Rotating and normalizing t-SNE coordinates...")
    theta = compute_rotation(bbox)
    rotated_tsne = rotate(tsne_results, origin=bbox[0], theta=theta)
    normalized_tsne = rotated_tsne - np.min(rotated_tsne, axis=0)
    print(f"Normalized t-SNE range: x=[0, {np.max(normalized_tsne[:,0]):.1f}], y=[0, {np.max(normalized_tsne[:,1]):.1f}]")
    
    # Visualize rotated bounding box
    rotated_bbox = rotate(bbox, origin=bbox[0], theta=theta)
    plot_bounding_box(rotated_tsne, rotated_bbox, output_path="loaded_data/tsne_rotated_bbox.png")
    
    # Step 5: Create expression images from t-SNE coordinates
    print("\n[5/8] Creating expression images from t-SNE coordinates...")
    data, w, h = initialize_image_data(sample_gene_expressions, normalized_tsne)
    print(f"Initial image dimensions: {w+1} x {h+1}")
    
    data = load_if_not_exists("loaded_data/unpadded_expressions.npy", 
        create_expression_images_from_tsne, 
        sample_gene_expressions=sample_gene_expressions,
        normalized_tsne=normalized_tsne,
        data=data,
        w=w,
        h=h)
    print(f"Expression images shape: {data.shape}")
    
    # Step 6: Pad images to target size
    print(f"\n[6/8] Padding images to {TARGET_SIZE}x{TARGET_SIZE}...")
    data = pad_data(data, TARGET_SIZE)
    
    # Step 7: Resize images (if needed)
    print(f"\n[7/8] Resizing images to {TARGET_SIZE}x{TARGET_SIZE}...")
    data = load_if_not_exists("loaded_data/resized_expressions.npy",
        resize_images,
        images=data,
        target_size=TARGET_SIZE)
    print(f"Final image shape: {data.shape}")
    print(f"Pixel value range: [{np.min(data):.4f}, {np.max(data):.4f}]")
    
    # Visualize sample images
    render_image(data[0], output_path="loaded_data/sample_image_0.png", title="Sample Expression Image 0")
    render_image(data[100], output_path="loaded_data/sample_image_100.png", title="Sample Expression Image 100")
    render_image(data[500], output_path="loaded_data/sample_image_500.png", title="Sample Expression Image 500")
    
    # Step 8: Prepare labels
    print("\n[8/8] Preparing phenotype labels...")
    samples = load_if_not_exists("loaded_data/samples.npy", load_samples)
    
    # Primary disease/tissue labels
    phenotype_mapping = load_if_not_exists("loaded_data/sample_to_body_site_mapping.json", 
        generate_phenotype_mapping)
    sample_body_site_phenotypes = load_if_not_exists("loaded_data/sample_body_site_phenotypes.npy", 
        get_phenotypes, 
        samples=samples, 
        sample_to_phenotype=phenotype_mapping)
    y_train_primary_disease_or_tissue = load_if_not_exists("loaded_data/y_primary_disease_or_tissue.npy",
        get_y_train,
        phenotypes=sample_body_site_phenotypes)
    print(f"Primary disease/tissue labels shape: {y_train_primary_disease_or_tissue.shape}")
    print(f"Number of classes: {y_train_primary_disease_or_tissue.shape[1]}")
    
    # Primary site labels (alternative classification target)
    primary_site_mapping = load_if_not_exists("loaded_data/primary_site_mapping.json", 
        generate_phenotype_mapping, 
        source_file=GTEX_PHENOTYPE, 
        target_column=2)
    sample_primary_site_phenotypes = load_if_not_exists("loaded_data/sample_primary_site_phenotypes.npy",
        get_phenotypes,
        samples=samples,
        sample_to_phenotype=primary_site_mapping)
    y_train_primary_site = load_if_not_exists("loaded_data/y_primary_site.npy",
        get_y_train,
        phenotypes=sample_primary_site_phenotypes)
    print(f"Primary site labels shape: {y_train_primary_site.shape}")
    print(f"Number of classes: {y_train_primary_site.shape[1]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTraining data prepared:")
    print(f"  - Images: {data.shape} (samples, height, width)")
    print(f"  - Labels (disease/tissue): {y_train_primary_disease_or_tissue.shape}")
    print(f"  - Labels (site): {y_train_primary_site.shape}")
    print(f"\nOutput files saved in loaded_data/")
    print(f"  Data files:")
    print(f"    - resized_expressions.npy")
    print(f"    - y_primary_disease_or_tissue.npy")
    print(f"    - y_primary_site.npy")
    print(f"  Visualization files:")
    print(f"    - tsne_scatter.png")
    print(f"    - tsne_bounding_box.png")
    print(f"    - tsne_rotated_bbox.png")
    print(f"    - tsne_convex_hull.png")
    print(f"    - sample_image_0.png")
    print(f"    - sample_image_100.png")
    print(f"    - sample_image_500.png")
    print("\nReady for model training with Classifier.py")
