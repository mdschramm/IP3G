#!/usr/bin/env python
"""
Fidelity analysis for diffusion-generated synthetic samples.

Three checks:
  1. intra_class_diversity  -- detects mode collapse: are synthetic samples
                               within a class all nearly identical?
  2. nearest_neighbor_dist  -- detects memorization: are synthetic samples
                               suspiciously close to specific training examples?
  3. distribution_fidelity  -- checks mean-image alignment: does the synthetic
                               distribution center match the real distribution center
                               per class?

Usage:
    python -m evaluation.diffusion_sample_fidelity_analysis [--mode all|diversity|memorization|fidelity]
    python -m evaluation.diffusion_sample_fidelity_analysis --guidance-scale 3.0

Output: evaluation/fidelity_analysis/
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PREPROCESSING_DIR = "output/preprocessing"
DIFFUSION_DIR     = "output/diffusion/diagnostic"
OUTPUT_DIR        = "evaluation/fidelity_analysis"

EXCLUDED_CLASSES  = [6, 24, 25, 31]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_real():
    X = np.load(f"{PREPROCESSING_DIR}/resized_expressions.npy").astype(np.float32)
    y = np.argmax(np.load(f"{PREPROCESSING_DIR}/y_primary_disease_or_tissue.npy"), axis=1)
    return X, y


def _load_synthetic(guidance_scale=3.0):
    sample_dir = os.path.join(DIFFUSION_DIR, "samples")
    X = np.load(os.path.join(sample_dir, f"diffusion_synthetic_expressions_w{guidance_scale:.1f}.npy")).astype(np.float32)
    y = np.argmax(np.load(os.path.join(sample_dir, f"diffusion_synthetic_labels_w{guidance_scale:.1f}.npy")), axis=1)
    print("synthetic samples:", X.shape, y.shape)
    return X, y


def _pairwise_l2(A, B):
    """L2 distance matrix between rows of A (m, d) and B (n, d). Returns (m, n)."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    a2 = (A ** 2).sum(axis=1, keepdims=True)   # (m, 1)
    b2 = (B ** 2).sum(axis=1)                   # (n,)
    return np.sqrt(np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0))


def _active_classes(y_synth, y_real):
    """Return sorted list of class indices present in both datasets and not excluded."""
    return sorted(
        c for c in np.unique(y_synth)
        if c not in EXCLUDED_CLASSES and np.sum(y_real == c) > 0
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intra-class diversity (mode collapse check)
# ─────────────────────────────────────────────────────────────────────────────

def intra_class_diversity(guidance_scale=3.0):
    """
    Check for mode collapse within generated samples for each class.

    For each non-excluded class computes two metrics:

      diversity_ratio = mean pixel-wise std of synthetic samples
                        ─────────────────────────────────────────
                        mean pixel-wise std of real training samples

        → ratio near 0 : all synthetic samples look identical (full collapse)
        → ratio near 1 : synthetic samples span the same variation as real data
        → ratio > 1    : synthetic samples are more spread than real (unlikely
                         but possible with heavy guidance)

      mean_pairwise_l2 : mean L2 distance between all C(n,2) pairs of synthetic
                         samples within the class. A near-zero value independently
                         confirms collapse.

    Saves: evaluation/fidelity_analysis/diversity.png
    Prints: per-class table sorted by diversity_ratio ascending (worst first).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_real, y_real = _load_real()
    X_synth, y_synth = _load_synthetic(guidance_scale)
    classes = _active_classes(y_synth, y_real)

    ratios, mean_pw_l2 = {}, {}

    for c in classes:
        synth_c = X_synth[y_synth == c]           # (n_synth, H, W)
        real_c  = X_real[y_real == c]             # (n_real,  H, W)

        synth_std = synth_c.std(axis=0).mean()    # scalar: mean pixel-wise std
        real_std  = real_c.std(axis=0).mean()

        ratios[c] = float(synth_std / real_std) if real_std > 1e-8 else 0.0

        # Mean pairwise L2 between synthetic samples
        flat = synth_c.reshape(len(synth_c), -1)
        if len(flat) >= 2:
            dists = _pairwise_l2(flat, flat)
            i_upper, j_upper = np.triu_indices(len(flat), k=1)
            mean_pw_l2[c] = float(dists[i_upper, j_upper].mean())
        else:
            mean_pw_l2[c] = 0.0

    # --- print table (worst first) ---
    print("\n" + "=" * 62)
    print("Intra-class diversity (mode collapse check)")
    print(f"  guidance_scale = {guidance_scale}")
    print(f"  {'Class':>6}  {'n_synth':>8}  {'n_real':>7}  "
          f"{'ratio':>8}  {'mean_pw_l2':>12}")
    print("  " + "-" * 52)
    for c in sorted(classes, key=lambda c: ratios[c]):
        flag = "  ← LOW" if ratios[c] < 0.3 else ""
        print(f"  {c:>6}  {int((y_synth == c).sum()):>8}  "
              f"{int((y_real == c).sum()):>7}  "
              f"{ratios[c]:>8.3f}  {mean_pw_l2[c]:>12.4f}{flag}")
    print(f"\n  Median ratio: {np.median(list(ratios.values())):.3f}  "
          f"  Min: {min(ratios.values()):.3f}  Max: {max(ratios.values()):.3f}")
    print("=" * 62)

    # --- plot ---
    x = np.array(sorted(classes))
    y_vals = np.array([ratios[c] for c in x])

    colors = ['tomato' if v < 0.3 else ('gold' if v < 0.6 else 'steelblue')
              for v in y_vals]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x, y_vals, color=colors, width=0.7)
    ax.axhline(1.0, color='black',   linestyle='--', linewidth=1.0, label='Ratio = 1 (matches real spread)')
    ax.axhline(0.3, color='tomato',  linestyle=':',  linewidth=1.2, label='Ratio = 0.3 (low diversity threshold)')
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=7)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Diversity ratio  (synth std / real std)")
    ax.set_title(
        f"Intra-class Diversity Ratio — guidance scale {guidance_scale}\n"
        "(red = low diversity / possible collapse; blue = healthy)"
    )
    ax.legend(fontsize=9)

    out_path = os.path.join(OUTPUT_DIR, f"diversity_w{guidance_scale:.1f}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Nearest-neighbour distance (memorization check)
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbor_dist(guidance_scale=3.0):
    """
    Check for memorization: compare how close synthetic samples are to real
    training examples versus how close real examples are to each other.

    For each non-excluded class computes:

      synth_nn : mean (over synthetic samples) of the L2 distance to the nearest
                 real training sample from the SAME class.
      real_nn  : mean (over real samples) of the L2 distance to the nearest OTHER
                 real sample from the same class (leave-one-out same-class NN).

    Interpretation:
      synth_nn << real_nn → synthetic samples collapse onto specific training
                            examples; strong sign of memorization.
      synth_nn ≈ real_nn  → synthetic samples sit on the same manifold as
                            training data; healthy.
      synth_nn >> real_nn → synthetic samples have drifted away from real data;
                            poor fidelity or out-of-distribution generation.

    Saves: evaluation/fidelity_analysis/memorization_w{g}.png
    Prints: per-class table with synth_nn, real_nn, and their ratio.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_real, y_real = _load_real()
    X_synth, y_synth = _load_synthetic(guidance_scale)
    classes = _active_classes(y_synth, y_real)

    synth_nns, real_nns = {}, {}

    for c in classes:
        flat_synth = X_synth[y_synth == c].reshape(int((y_synth == c).sum()), -1)
        flat_real  = X_real[y_real == c].reshape(int((y_real == c).sum()), -1)

        # synth → real same-class NN
        d_sr = _pairwise_l2(flat_synth, flat_real)         # (n_synth, n_real)
        synth_nns[c] = float(d_sr.min(axis=1).mean())

        # real → real same-class NN (leave-one-out: exclude self via masking diagonal)
        d_rr = _pairwise_l2(flat_real, flat_real)          # (n_real, n_real)
        np.fill_diagonal(d_rr, np.inf)
        real_nns[c] = float(d_rr.min(axis=1).mean())

    # --- print table ---
    print("\n" + "=" * 68)
    print("Nearest-neighbour distances (memorization check)")
    print(f"  guidance_scale = {guidance_scale}")
    print(f"  {'Class':>6}  {'synth_nn':>10}  {'real_nn':>10}  "
          f"{'ratio s/r':>10}  Note")
    print("  " + "-" * 56)
    for c in sorted(classes, key=lambda c: synth_nns[c] / real_nns[c] if real_nns[c] > 0 else 0):
        ratio = synth_nns[c] / real_nns[c] if real_nns[c] > 1e-8 else float('inf')
        if ratio < 0.5:
            note = "← possible memorization"
        elif ratio > 2.0:
            note = "← out-of-distribution"
        else:
            note = ""
        print(f"  {c:>6}  {synth_nns[c]:>10.4f}  {real_nns[c]:>10.4f}  "
              f"{ratio:>10.3f}  {note}")
    print("=" * 68)

    # --- plot ---
    x = np.array(sorted(classes))
    s_vals = np.array([synth_nns[c] for c in x])
    r_vals = np.array([real_nns[c]  for c in x])

    width = 0.4
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width / 2, s_vals, width, label='Synthetic → real NN (synth_nn)', color='steelblue', alpha=0.85)
    ax.bar(x + width / 2, r_vals, width, label='Real → real NN baseline (real_nn)',   color='darkorange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=7)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Mean nearest-neighbour L2 distance")
    ax.set_title(
        f"Nearest-Neighbour Distance — guidance scale {guidance_scale}\n"
        "(blue < orange → synthetic samples closer to training data than training is to itself → memorization risk)"
    )
    ax.legend(fontsize=9)

    out_path = os.path.join(OUTPUT_DIR, f"memorization_w{guidance_scale:.1f}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Distribution fidelity (mean-image alignment)
# ─────────────────────────────────────────────────────────────────────────────

def distribution_fidelity(guidance_scale=3.0):
    """
    Check whether the synthetic samples are centered on the correct class distribution.

    For each non-excluded class computes:

      mean_l2 : L2 distance between the mean synthetic image and the mean real
                image for that class (pixel space).

                A small mean_l2 means the model has learned the right average
                expression pattern for that class. A large value means the model
                is generating images centered somewhere else in the image space —
                a fidelity failure independent of mode collapse.

      std_ratio : mean pixel-wise std of synthetic samples divided by mean
                  pixel-wise std of real samples (same as intra_class_diversity).
                  Included here alongside mean_l2 so both aspects of distributional
                  match (center and spread) can be read together.

    Saves: evaluation/fidelity_analysis/fidelity_w{g}.png  (two-panel plot)
    Prints: per-class table sorted by mean_l2 descending (worst first).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_real, y_real = _load_real()
    X_synth, y_synth = _load_synthetic(guidance_scale)
    classes = _active_classes(y_synth, y_real)

    mean_l2s, mean_maes, std_ratios = {}, {}, {}

    for c in classes:
        synth_c = X_synth[y_synth == c].astype(np.float64)
        real_c  = X_real[y_real == c].astype(np.float64)

        mean_synth = synth_c.mean(axis=0)
        mean_real  = real_c.mean(axis=0)
        diff = mean_synth - mean_real
        n_pixels = diff.size
        mean_l2s[c]  = float(np.linalg.norm(diff))
        # Normalize by sqrt(n_pixels) so the value is comparable across image sizes
        # and interpretable as an RMS per-pixel difference in [0, 1]
        mean_maes[c] = float(mean_l2s[c] / np.sqrt(n_pixels))

        synth_std = synth_c.std(axis=0).mean()
        real_std  = real_c.std(axis=0).mean()
        std_ratios[c] = float(synth_std / real_std) if real_std > 1e-8 else 0.0

    # --- print table ---
    # Use normalised L2 (RMS per-pixel) for the 90th-pct threshold — it's in [0,1]
    p90_mae = float(np.percentile(list(mean_maes.values()), 90))
    print("\n" + "=" * 72)
    print("Distribution fidelity (mean-image alignment)")
    print(f"  guidance_scale = {guidance_scale}")
    print(f"  mean_l2     = raw L2 distance (scales with sqrt(H*W) = {int(np.sqrt(n_pixels))})")
    print(f"  rms_per_px  = mean_l2 / sqrt(H*W)  — RMS per-pixel difference, in [0, 1]")
    print(f"  {'Class':>6}  {'mean_l2':>10}  {'rms_per_px':>12}  {'std_ratio':>10}  Note")
    print("  " + "-" * 58)
    for c in sorted(classes, key=lambda c: -mean_maes[c]):
        note = "← high center drift" if mean_maes[c] > p90_mae else ""
        print(f"  {c:>6}  {mean_l2s[c]:>10.4f}  {mean_maes[c]:>12.5f}  "
              f"{std_ratios[c]:>10.3f}  {note}")
    print(f"\n  Median rms_per_px:  {np.median(list(mean_maes.values())):.5f}")
    print(f"  90th-pct:           {p90_mae:.5f}")
    print("=" * 72)

    # --- two-panel plot (use rms_per_px for panel 1) ---
    x = np.array(sorted(classes))
    mae_vals = np.array([mean_maes[c]  for c in x])
    std_vals = np.array([std_ratios[c] for c in x])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Panel 1: normalised mean-image distance
    colors1 = ['tomato' if v > p90_mae else 'steelblue' for v in mae_vals]
    ax1.bar(x, mae_vals, color=colors1, width=0.7)
    ax1.axhline(p90_mae, color='tomato', linestyle=':', linewidth=1.2,
                label=f'90th percentile ({p90_mae:.5f})')
    ax1.set_ylabel("RMS per-pixel difference  (mean_l2 / √pixels)")
    ax1.set_title(
        f"Distribution Fidelity — guidance scale {guidance_scale}\n"
        "Panel 1: RMS per-pixel difference between synthetic and real mean image  (lower = better, range [0,1])"
    )
    ax1.legend(fontsize=9)

    # Panel 2: std ratio
    colors2 = ['tomato' if v < 0.3 else ('gold' if v < 0.6 else 'steelblue')
               for v in std_vals]
    ax2.bar(x, std_vals, color=colors2, width=0.7)
    ax2.axhline(1.0, color='black',  linestyle='--', linewidth=1.0, label='Ratio = 1 (matches real spread)')
    ax2.axhline(0.3, color='tomato', linestyle=':',  linewidth=1.2, label='Low diversity threshold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x, fontsize=7)
    ax2.set_xlabel("Class index")
    ax2.set_ylabel("Std ratio  (synth / real)")
    ax2.set_title("Panel 2: Std ratio  (lower = less spread than real data)")
    ax2.legend(fontsize=9)

    out_path = os.path.join(OUTPUT_DIR, f"fidelity_w{guidance_scale:.1f}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fidelity analysis for diffusion-generated synthetic samples"
    )
    parser.add_argument(
        '--mode',
        choices=['all', 'diversity', 'memorization', 'fidelity'],
        default='all',
        help=(
            '"diversity"     — intra-class diversity / mode collapse check; '
            '"memorization"  — nearest-neighbour distance memorization check; '
            '"fidelity"      — mean-image alignment and std ratio; '
            '"all"           — run all three (default)'
        )
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=3.0,
        help='Guidance scale used when generating samples (determines input filenames, default: 3.0)'
    )
    args = parser.parse_args()
    g = args.guidance_scale

    if args.mode in ('all', 'diversity'):
        intra_class_diversity(g)
    if args.mode in ('all', 'memorization'):
        nearest_neighbor_dist(g)
    if args.mode in ('all', 'fidelity'):
        distribution_fidelity(g)
