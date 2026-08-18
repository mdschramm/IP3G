#!/usr/bin/env python
"""
Main script to prepare training data from raw gene expression files.

This script orchestrates the full preprocessing pipeline:
1. Load raw gene expression data
2. Apply t-SNE dimensionality reduction
3. Compute minimum bounding rectangle
4. Rotate, normalize, and scale t-SNE coordinates
5. Load phenotypes + compute per-gene ANOVA F-statistic importance order
6. Create 16-channel multichannel expression images from t-SNE coordinates
7. Pad images to 128×128
7.5. Compute sigma_data (global and occupied) for diffusion config calibration
8. Prepare one-hot encoded labels

OUTPUT FILES:
- output/preprocessing/resized_expressions.npy:         (N, 128, 128, 16) float32 images
- output/preprocessing/pixel_occupancy_mask.npy:        (128, 128, 16) bool
- output/preprocessing/gene_pixel_channel.npy:          (N_genes, 3) int32
- output/preprocessing/channel_scales.npy:              (16,) float32
- output/preprocessing/gene_importance_order.npy:       (N_genes,) int64
- output/preprocessing/gene_f_stats.npy:                (N_genes,) float32
- output/preprocessing/sigma_data.json:                 {global, occupied}
- output/preprocessing/y_primary_disease_or_tissue.npy: (N, N_classes) one-hot
- output/preprocessing/y_primary_site.npy:              (N, N_classes) one-hot

USAGE:
    python -m preprocessing.prepare_training_data

The script uses caching (load_if_not_exists) so intermediate results are saved.
To force recalculation, delete the cached .npy files in output/preprocessing/

NOTE: t-SNE coordinates are uniformly scaled to fit within 127×127 pixels before
image creation, so pad_data is the only spatial step needed — no interpolation.
Pixel values are exactly [0, 1] per channel in the output.
"""

import json
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
    compute_gene_importance_order,
    create_multichannel_expression_images_from_tsne,
    pad_data,
    TARGET_SIZE
)

# Visualization functions (save to PNG files)
from preprocessing.visualization import plot_tsne, plot_bounding_box, plot_convex_hull
from scipy.spatial import ConvexHull


if __name__ == "__main__":
    import os
    OUT = "output/preprocessing"
    os.makedirs(OUT, exist_ok=True)

    print("=" * 80)
    print("GENE EXPRESSION TO IMAGE PREPROCESSING PIPELINE (16-channel)")
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
    scale = (TARGET_SIZE - 1) / np.max(normalized_tsne)
    normalized_tsne = normalized_tsne * scale
    print(f"  Scaled by {scale:.4f} → coordinates fit within {TARGET_SIZE-1}×{TARGET_SIZE-1} px")

    rotated_bbox = rotate(bbox, origin=bbox[0], theta=theta)
    plot_bounding_box(rotated_tsne, rotated_bbox, output_path=f"{OUT}/tsne_rotated_bbox.png")

    # Compute w, h (needed for multichannel image creation)
    w, h = np.max(normalized_tsne, axis=0)
    w, h = int(w), int(h)
    print(f"  Image dimensions (before padding): {w+1} x {h+1}")

    print("\n[5/8] Loading phenotypes and computing gene importance order...")
    samples = load_if_not_exists(f"{OUT}/samples.npy", load_samples)
    phenotype_mapping = load_if_not_exists(f"{OUT}/sample_to_body_site_mapping.json",
        generate_phenotype_mapping)
    sample_body_site_phenotypes = load_if_not_exists(f"{OUT}/sample_body_site_phenotypes.npy",
        get_phenotypes,
        samples=samples,
        sample_to_phenotype=phenotype_mapping)
    print(f"  Phenotypes loaded: {len(sample_body_site_phenotypes)} samples, "
          f"{len(set(sample_body_site_phenotypes))} classes")

    def _compute_importance(**kwargs):
        print("  Computing ANOVA F-statistic per gene (this may take a few minutes)...")
        order, f_stats = compute_gene_importance_order(
            kwargs['sample_gene_expressions'], kwargs['sample_body_site_phenotypes']
        )
        np.save(f"{OUT}/gene_f_stats.npy", f_stats)
        print(f"  Saved: gene_importance_order.npy, gene_f_stats.npy")
        return order

    gene_importance_order = load_if_not_exists(
        f"{OUT}/gene_importance_order.npy",
        _compute_importance,
        sample_gene_expressions=sample_gene_expressions,
        sample_body_site_phenotypes=sample_body_site_phenotypes,
    )

    print("\n[6/8] Creating 16-channel expression images from t-SNE coordinates...")

    def _create_multichannel(**kwargs):
        # Wrapper so load_if_not_exists can cache only the image array;
        # occupancy mask and reverse-map tables are saved separately here.
        data, pom, gpc, cs = create_multichannel_expression_images_from_tsne(
            sample_gene_expressions=kwargs['sample_gene_expressions'],
            normalized_tsne=kwargs['normalized_tsne'],
            gene_importance_order=kwargs['gene_importance_order'],
            w=kwargs['w'],
            h=kwargs['h'],
        )
        # Save the side-channel artifacts immediately (not cached by load_if_not_exists)
        np.save(f"{OUT}/pixel_occupancy_mask.npy", pom)
        np.save(f"{OUT}/gene_pixel_channel.npy", gpc)
        np.save(f"{OUT}/channel_scales.npy", cs)
        print(f"  Saved: pixel_occupancy_mask.npy  gene_pixel_channel.npy  channel_scales.npy")
        print(f"  Occupancy mask shape: {pom.shape}  occupied: {pom.sum():,} / {pom.size:,} positions")
        print(f"  channel_scales (first 8): {cs[:8]}")
        return data

    data = load_if_not_exists(f"{OUT}/unpadded_expressions.npy",
        _create_multichannel,
        sample_gene_expressions=sample_gene_expressions,
        normalized_tsne=normalized_tsne,
        gene_importance_order=gene_importance_order,
        w=w,
        h=h)
    print(f"  Unpadded expression images shape: {data.shape}")

    print(f"\n[7/8] Padding images to {TARGET_SIZE}×{TARGET_SIZE}...")
    data = pad_data(data, TARGET_SIZE)
    print(f"  Padded shape: {data.shape}")

    # pad_data center-pads: gene (px, py) in the unpadded image lands at
    # (px + left_padd, py + top_padd) in resized_expressions.npy.
    # Both gene_pixel_channel and pixel_occupancy_mask must reflect that offset.
    left_padd = (TARGET_SIZE - (w + 1)) // 2
    top_padd  = (TARGET_SIZE - (h + 1)) // 2

    # Update gene_pixel_channel spatial coords to padded coordinate space.
    # Idempotent: if max x-coord already exceeds w, the shift was already applied.
    gpc = np.load(f"{OUT}/gene_pixel_channel.npy")
    if int(gpc[:, 0].max()) <= w:
        gpc[:, 0] += left_padd
        gpc[:, 1] += top_padd
        np.save(f"{OUT}/gene_pixel_channel.npy", gpc)
        print(f"  Updated gene_pixel_channel: shifted coords by left={left_padd}, top={top_padd}")

    # Pad the occupancy mask to match — it was saved with the pre-padding spatial dims.
    # pad_data expects (N, H, W, C); wrap with [None] / unwrap with [0].
    occ_mask_unpadded = np.load(f"{OUT}/pixel_occupancy_mask.npy")
    if occ_mask_unpadded.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
        occ_mask_padded = pad_data(occ_mask_unpadded[None], TARGET_SIZE)[0]
        np.save(f"{OUT}/pixel_occupancy_mask.npy", occ_mask_padded)
        print(f"  Padded occupancy mask: {occ_mask_unpadded.shape} → {occ_mask_padded.shape}")

    out_path = f"{OUT}/resized_expressions.npy"
    if not os.path.exists(out_path):
        np.save(out_path, data.astype(np.float32))
        print(f"  Saved: {out_path}")
    else:
        data = np.load(out_path)
        print(f"  Loaded from cache: {out_path}")
    print(f"  Final image shape: {data.shape}")
    print(f"  Pixel value range: [{np.min(data):.4f}, {np.max(data):.4f}]")

    print("\n[7.5/8] Computing sigma_data for diffusion config calibration...")
    occ_mask_path = f"{OUT}/pixel_occupancy_mask.npy"
    sigma_data_path = f"{OUT}/sigma_data.json"
    if not os.path.exists(sigma_data_path):
        if os.path.exists(occ_mask_path):
            occ_mask = np.load(occ_mask_path)          # (128, 128, 16) bool
            sigma_data_global   = float(np.std(data))
            sigma_data_occupied = float(np.std(data[:, occ_mask]))   # occupied positions only
            sigma_payload = {"global": sigma_data_global, "occupied": sigma_data_occupied}
            with open(sigma_data_path, "w") as f:
                json.dump(sigma_payload, f)
            print(f"  sigma_data global={sigma_data_global:.4f}  occupied={sigma_data_occupied:.4f}")
            print(f"  → Use sigma_data_occupied={sigma_data_occupied:.4f} in diffusion configs")
            print(f"  → Set P_mean={np.log(sigma_data_occupied):.4f} (= ln(sigma_data_occupied))")
            print(f"  Saved: {sigma_data_path}")
        else:
            print("  SKIP: pixel_occupancy_mask.npy not found (run without cache to generate it)")
    else:
        with open(sigma_data_path) as f:
            sd = json.load(f)
        print(f"  Loaded from cache: sigma_data global={sd['global']:.4f}  occupied={sd['occupied']:.4f}")

    print("\n[8/8] Preparing phenotype labels (one-hot)...")
    y_train_primary_disease_or_tissue = load_if_not_exists(f"{OUT}/y_primary_disease_or_tissue.npy",
        get_y_train,
        phenotypes=sample_body_site_phenotypes)
    print(f"  Primary disease/tissue labels shape: {y_train_primary_disease_or_tissue.shape}")
    print(f"  Number of classes: {y_train_primary_disease_or_tissue.shape[1]}")

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
    print(f"  Primary site labels shape: {y_train_primary_site.shape}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTraining data prepared:")
    print(f"  - Images:                   {data.shape}  (N, H, W, C)")
    print(f"  - Labels (disease/tissue):  {y_train_primary_disease_or_tissue.shape}")
    print(f"  - Labels (site):            {y_train_primary_site.shape}")
    print(f"\nOutput files saved in {OUT}/")
    print(f"  resized_expressions.npy, pixel_occupancy_mask.npy,")
    print(f"  gene_pixel_channel.npy, channel_scales.npy,")
    print(f"  gene_importance_order.npy, gene_f_stats.npy, sigma_data.json,")
    print(f"  y_primary_disease_or_tissue.npy, y_primary_site.npy")
    print("\nReady for model training.")
