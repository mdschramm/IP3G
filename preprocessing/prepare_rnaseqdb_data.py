#!/usr/bin/env python
"""
Prepare training data from the combined TCGA + GTEx corpus (RNAseqDB).

The RNAseqDB counterpart to prepare_training_data.py. Stages 1-4 are specific to
this corpus; stages 5-8 reuse preprocessing/image_preprocessing.py unchanged.

STAGES
  1. Fetch the 52 normalized matrices           (rnaseqdb_loader.download_rnaseqdb)
  2. Intersect gene lists across all 52 files   (rnaseqdb_loader.gene_intersection)
  3. Join, log2(FPKM+1), clip                   (rnaseqdb_loader.load_rnaseqdb)
  4. Build the label frame and encode it        (label_frame)
  5. t-SNE over genes                           (image_preprocessing, shared)
  6. Tissue F-statistic gene ordering           (image_preprocessing, shared)
  7. Multichannel image mapping                 (image_preprocessing, shared)
  8. sigma_data for the diffusion config

OUTPUT — all under output/preprocessing/rnaseqdb/, see artifact_paths.py:
    data.npy                (9147, 18154) float32   log2(FPKM+1), clipped
    gene_symbols.npy        (18154,)      <U        Hugo symbols
    gene_entrez.npy         (18154,)      int64     Entrez IDs
    samples.npy             (9147,)       <U        GTEx SAMPIDs / TCGA barcodes
    labels.csv              (9147, 5)               sample_id + four attributes
    attribute_vocab.json                            index -> value per attribute
    y_tissue.npy            (9147, 15)    float32
    y_condition.npy         (9147, 2)     float32
    y_subtype.npy           (9147, 20)    float32
    y_source.npy            (9147, 2)     float32
    tsne_results.npy        (18154, 2)    float32
    gene_importance_order.npy / gene_f_stats.npy    (18154,)
    resized_expressions.npy (9147, W, H, C) float32 in [0, 1]
    pixel_occupancy_mask.npy / gene_pixel_channel.npy / channel_scales.npy
    sigma_data.json

NOTE ON PADDING — differs from the GTEx pipeline, identical in result.
    prepare_training_data.py builds images at the t-SNE's natural extent and then
    calls pad_data to centre them in the target square, which allocates a second
    full-size array: 9.6 GB + 9.6 GB = 19.2 GB peak here, enough to thrash a 16 GB
    machine. Instead we add the centring offsets to the t-SNE coordinates *before*
    image creation and build directly at the target size, so nothing is copied.
    Since the offsets are integers and coordinates are non-negative,
    int(x + k) == int(x) + k, so gene placement is bit-identical either way — and
    gene_pixel_channel comes out already in padded coordinates, with no post-hoc
    shift needed.

USAGE
    python -m preprocessing.prepare_rnaseqdb_data
    python -m preprocessing.prepare_rnaseqdb_data --force        # ignore caches
    python -m preprocessing.prepare_rnaseqdb_data --no-download  # use local files only
    python -m preprocessing.prepare_rnaseqdb_data --stages 1-4   # labels only
    python -m preprocessing.prepare_rnaseqdb_data --width 256 --height 256 --channels 1

Caching follows the GTEx pipeline: existing artifacts are reused rather than
rebuilt. Delete the artifact directory to force a clean run.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from preprocessing.artifact_paths import PreprocessingConfig, RNASEQDB_CONFIG
from preprocessing.image_preprocessing import (
    compute_gene_importance_order,
    compute_rotation,
    create_multichannel_expression_images_from_tsne,
    get_tsne_data,
    minimum_bounding_rectangle,
    rotate,
)
from preprocessing.label_frame import (
    attribute_sizes,
    build_vocab,
    describe,
    one_hot,
    save_vocab,
)
from preprocessing.preprocess_data import load_if_not_exists
from preprocessing.rnaseqdb_loader import (
    ATTRIBUTES,
    DEFAULT_DATA_DIR,
    EXPECTED_GENES,
    EXPECTED_SAMPLES,
    LOG2_CLIP,
    load_rnaseqdb,
)

# Which attribute the gene-importance F-statistic ranks against. Tissue, matching
# the GTEx pipeline: at 16 channels the ranking governs ~0.1% of genes, so this is
# a low-stakes choice until the channel budget drops to 4 or fewer.
IMPORTANCE_ATTRIBUTE = "tissue"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RNAseqDB (TCGA + GTEx) training data"
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="directory holding the 52 *.txt.gz matrices")
    parser.add_argument("--clip", type=float, default=LOG2_CLIP,
                        help="log2(FPKM+1) ceiling; 0 disables clipping")
    parser.add_argument("--no-download", action="store_true",
                        help="never fetch from GitHub; fail if files are missing")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when cached artifacts exist")
    parser.add_argument("--stages", default="1-8",
                        help="'1-4' for labels only, '1-8' for the full pipeline")
    parser.add_argument("--width", type=int, default=RNASEQDB_CONFIG.width)
    parser.add_argument("--height", type=int, default=RNASEQDB_CONFIG.height)
    parser.add_argument("--channels", type=int, default=RNASEQDB_CONFIG.channels)
    args = parser.parse_args()

    last_stage = int(args.stages.split("-")[-1])
    config = PreprocessingConfig(width=args.width, height=args.height,
                                 channels=args.channels, dataset="rnaseqdb")
    assert config.width == config.height, "Pipeline assumes a square target image"
    TARGET_SIZE = config.width

    out = config.dataset_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(config.artifact_dir, exist_ok=True)

    print("=" * 78)
    print(f"RNASEQDB (TCGA + GTEx) PREPROCESSING — stages {args.stages} — {config.tag}")
    print("=" * 78)
    print(f"  source files : {args.data_dir}")
    print(f"  artifacts    : {out}")

    # ---- stages 1-3 -------------------------------------------------------
    if os.path.exists(config.data_path) and not args.force:
        print(f"\n[1-3] Cached: {config.data_path}")
        X = np.load(config.data_path)
        genes = pd.DataFrame({
            "Hugo_Symbol": np.load(config.gene_symbols_path, allow_pickle=True),
            "Entrez_Gene_Id": np.load(config.gene_entrez_path),
        })
        labels = pd.read_csv(config.labels_path)
        print(f"      X {X.shape}  genes {len(genes)}  labels {labels.shape}")
    else:
        X, genes, labels = load_rnaseqdb(
            data_dir=args.data_dir,
            clip=args.clip if args.clip > 0 else None,
            download=not args.no_download,
        )
        np.save(config.data_path, X)
        np.save(config.gene_symbols_path, genes["Hugo_Symbol"].to_numpy().astype(str))
        np.save(config.gene_entrez_path, genes["Entrez_Gene_Id"].to_numpy(dtype=np.int64))
        np.save(config.samples_path, labels["sample_id"].to_numpy().astype(str))
        labels.to_csv(config.labels_path, index=False)
        print(f"\n  Saved: data.npy, gene_symbols.npy, gene_entrez.npy, "
              f"samples.npy, labels.csv")

    # ---- stage 4 ----------------------------------------------------------
    print("\n[4] Encoding label attributes...")
    vocab = build_vocab(labels, ATTRIBUTES)
    save_vocab(vocab, config.attribute_vocab_path)

    for attr in ATTRIBUTES:
        y = one_hot(labels, attr, vocab)
        np.save(config.y_attribute_path(attr), y)
        print(f"    y_{attr}.npy  {y.shape}")

    print("\n" + describe(labels, ATTRIBUTES, vocab))

    # ---- stages 5-8 -------------------------------------------------------
    images = None
    if last_stage >= 5:
        print(f"\n[5] t-SNE over {X.shape[1]} genes (this is the long pole)...")
        tsne_results = load_if_not_exists(config.tsne_results_path, get_tsne_data, data=X)
        print(f"    tsne_results {tsne_results.shape}")

        print("\n[6] Rotating / scaling coordinates and ranking genes...")
        bbox = minimum_bounding_rectangle(tsne_results)
        theta = compute_rotation(bbox)
        coords = rotate(tsne_results, origin=bbox[0], theta=theta)
        coords = coords - np.min(coords, axis=0)
        coords = coords * ((TARGET_SIZE - 1) / np.max(coords))
        w, h = (int(v) for v in np.max(coords, axis=0))
        print(f"    natural extent {w+1} x {h+1} px")

        # Centre the point cloud by shifting coordinates rather than padding the
        # image array afterwards — see NOTE ON PADDING above.
        left_pad = (TARGET_SIZE - (w + 1)) // 2
        top_pad = (TARGET_SIZE - (h + 1)) // 2
        coords = coords + np.array([left_pad, top_pad], dtype=coords.dtype)
        print(f"    pre-shifted by left={left_pad}, top={top_pad} -> "
              f"building directly at {TARGET_SIZE}x{TARGET_SIZE}")

        def _importance(**kw):
            order, f_stats = compute_gene_importance_order(
                kw["sample_gene_expressions"], kw["phenotypes"])
            np.save(config.gene_f_stats_path, f_stats)
            return order

        gene_order = load_if_not_exists(
            config.gene_importance_order_path, _importance,
            sample_gene_expressions=X,
            phenotypes=labels[IMPORTANCE_ATTRIBUTE].to_numpy(),
        )

        print(f"\n[7] Building {config.channels}-channel images...")

        def _create(**kw):
            data_, pom, gpc, cs = create_multichannel_expression_images_from_tsne(
                sample_gene_expressions=X,
                normalized_tsne=coords,
                gene_importance_order=gene_order,
                w=TARGET_SIZE - 1,
                h=TARGET_SIZE - 1,
                n_channels=config.channels,
            )
            np.save(config.pixel_occupancy_mask_path, pom)
            np.save(config.gene_pixel_channel_path, gpc)
            np.save(config.channel_scales_path, cs)
            return data_

        images = load_if_not_exists(config.resized_expressions_path, _create)
        print(f"    images {images.shape}  range [{images.min():.4f}, {images.max():.4f}]")

        print("\n[8] Computing sigma_data...")
        if not os.path.exists(config.sigma_data_path) or args.force:
            occ = np.load(config.pixel_occupancy_mask_path)
            payload = {"global": float(np.std(images)),
                       "occupied": float(np.std(images[:, occ]))}
            with open(config.sigma_data_path, "w") as fh:
                json.dump(payload, fh)
        with open(config.sigma_data_path) as fh:
            sigma = json.load(fh)
        print(f"    sigma_data global={sigma['global']:.4f} occupied={sigma['occupied']:.4f}")
        print(f"    -> diffusion config: sigma_data={sigma['occupied']:.4f}, "
              f"P_mean={np.log(sigma['occupied']):.4f}")

    # ---- M1 gate ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(("M1+M2" if images is not None else "M1") + " GATE")
    print("=" * 78)
    checks = [
        ("expression shape",
         X.shape == (EXPECTED_SAMPLES, EXPECTED_GENES),
         f"{X.shape} == ({EXPECTED_SAMPLES}, {EXPECTED_GENES})"),
        ("dtype float32", X.dtype == np.float32, str(X.dtype)),
        ("no NaN / inf", bool(np.isfinite(X).all()), "all finite"),
        ("values within clip",
         float(X.min()) >= 0.0 and float(X.max()) <= args.clip if args.clip > 0 else True,
         f"[{X.min():.3f}, {X.max():.3f}]"),
        ("gene axis aligned", len(genes) == X.shape[1], f"{len(genes)} genes"),
        ("sample axis aligned", len(labels) == X.shape[0], f"{len(labels)} labels"),
        ("sample ids unique", not labels["sample_id"].duplicated().any(), "no duplicates"),
        ("15 tissues", len(vocab["tissue"]) == 15, f"{len(vocab['tissue'])}"),
        ("2 conditions", len(vocab["condition"]) == 2, f"{len(vocab['condition'])}"),
        ("20 subtypes", len(vocab["subtype"]) == 20, f"{len(vocab['subtype'])}"),
        ("2 sources", len(vocab["source"]) == 2, f"{len(vocab['source'])}"),
    ]
    if images is not None:
        gpc = np.load(config.gene_pixel_channel_path)
        occ = np.load(config.pixel_occupancy_mask_path)
        cs = np.load(config.channel_scales_path)
        overflow = float((gpc[:, 2] == config.channels - 1).mean())
        checks += [
            ("image shape",
             images.shape == (EXPECTED_SAMPLES,) + config.image_shape,
             f"{images.shape}"),
            ("images in [0,1]",
             float(images.min()) >= 0.0 and float(images.max()) <= 1.0,
             f"[{images.min():.4f}, {images.max():.4f}]"),
            ("coords inside frame",
             int(gpc[:, 0].max()) < config.width and int(gpc[:, 1].max()) < config.height,
             f"max px={gpc[:, 0].max()}, py={gpc[:, 1].max()}"),
            ("every gene placed", len(gpc) == X.shape[1], f"{len(gpc)}"),
            ("occupancy mask shape", occ.shape == config.image_shape, f"{occ.shape}"),
            ("channel_scales positive",
             bool((cs > 0).all()), f"min={cs.min():.4g}"),
            ("overflow <= 2%", overflow <= 0.02, f"{100*overflow:.2f}% of genes"),
        ]

    ok = True
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<22} {detail}")

    print("\n  tissue x condition:")
    pivot = pd.crosstab(labels["tissue"], labels["condition"])
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.sort_values("total", ascending=False).to_string())
    print(f"\n  totals: {pivot['total'].sum()} samples "
          f"({(labels['condition'] == 'normal').sum()} normal, "
          f"{(labels['condition'] == 'tumor').sum()} tumor)")
    print(f"  attribute_sizes for the diffusion config: "
          f"{attribute_sizes(vocab, ('tissue', 'condition', 'subtype'))}")

    label = "M1+M2" if images is not None else "M1"
    print("\n" + (f"{label} COMPLETE" if ok else f"{label} GATE FAILED") + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
