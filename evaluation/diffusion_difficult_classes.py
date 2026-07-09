#!/usr/bin/env python
"""
Diagnostic analysis for the 4 difficult-to-classify brain sub-regions:
  Class  8: Brain - Anterior cingulate cortex (BA24)
  Class  9: Brain - Caudate (basal ganglia)
  Class 10: Brain - Cerebellar Hemisphere
  Class 13: Brain - Frontal Cortex (BA9)

Three analyses:
  1. centroid_correlation_stats     -- print table comparing confused-pair Pearson r
                                       to dataset-wide distribution statistics
  2. softmax_entropy                -- entropy of classifier softmax on real samples
  3. within_between_class_distance  -- intra-class spread vs inter-centroid L2 distance

Usage:
    python -m evaluation.diffusion_difficult_classes [--mode all|correlation|entropy|distance]

Output: evaluation/difficult_classes/  (analyses 2-4 save PNGs; analysis 1 is print-only)

NOTE on Pearson r between centroid images:
  Each class centroid is the pixel-wise mean of all training images for that class,
  flattened to a 1-D vector of length H*W (128*128 = 16384). Pearson r between two
  such vectors measures the linear similarity of their spatial expression patterns.
  Because gene expression images share a common sparse layout (most pixels are 0
  regardless of tissue type), the baseline correlation across all class pairs is
  already high (≥0.75). The confused pairs sitting near r=0.999 shows they are
  essentially indistinguishable at the mean-image level.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from classifer.Classifier import f1_m, precision_m, recall_m

PREPROCESSING_DIR  = "output/preprocessing"
CLASSIFIER_DIR     = "output/classifier/remote"
CLASSIFIER_WEIGHTS = "classifier_weights_only.keras"
OUTPUT_DIR         = "evaluation/difficult_classes"

DIFFICULT_CLASSES = [8, 9, 10, 13]

# Each tuple: (difficult_class, class_it_gets_confused_with)
CONFUSED_PAIRS = [(8, 13), (9, 17), (10, 11), (13, 12)]


def _load_real_data():
    X = np.load(f"{PREPROCESSING_DIR}/resized_expressions.npy").astype(np.float32)
    y = np.argmax(np.load(f"{PREPROCESSING_DIR}/y_primary_disease_or_tissue.npy"), axis=1)
    return X, y


def _load_classifier():
    tf.config.set_visible_devices([], 'GPU')
    weights_path = f"{CLASSIFIER_DIR}/{CLASSIFIER_WEIGHTS}"
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m}
    )
    print(f"Loaded classifier from {weights_path}")
    return model


def centroid_correlation_stats():
    """
    Compute Pearson r between the mean image (centroid) of every class pair, then
    print a table comparing the confused-pair values to dataset-wide statistics.

    Each centroid is the pixel-wise mean of all training images for that class,
    flattened to H*W = 16384 values. All gene expression images are sparse (mostly
    zero), so the baseline r across all pairs is already high. The table shows where
    the confused pairs rank relative to the full distribution.

    Also prints the top-10 most-correlated pairs across all classes so you can see
    whether the confused pairs are uniquely similar or part of a broader cluster.
    """
    X, y = _load_real_data()
    num_classes = int(y.max()) + 1

    centroids = np.stack([X[y == c].mean(axis=0).ravel() for c in range(num_classes)])
    corr = np.corrcoef(centroids)  # (num_classes, num_classes)

    # Upper triangle, excluding the diagonal (self-correlations of 1.0)
    i_upper, j_upper = np.triu_indices(num_classes, k=1)
    all_r = corr[i_upper, j_upper]

    print("\n" + "=" * 55)
    print("Centroid Pearson r — dataset-wide statistics")
    print(f"  All class pairs: n = {len(all_r)}")
    print(f"  Min:    {all_r.min():.4f}")
    print(f"  25th:   {np.percentile(all_r, 25):.4f}")
    print(f"  Median: {np.median(all_r):.4f}")
    print(f"  75th:   {np.percentile(all_r, 75):.4f}")
    print(f"  95th:   {np.percentile(all_r, 95):.4f}")
    print(f"  Max:    {all_r.max():.4f}")
    print(f"  Mean:   {all_r.mean():.4f}")

    print("\nConfused pair correlations vs. dataset-wide distribution:")
    print(f"  {'Pair':<12}  {'r':>8}  {'Percentile':>12}  Note")
    print("  " + "-" * 56)
    for a, b in CONFUSED_PAIRS:
        r = float(corr[a, b])
        pct = float(np.mean(all_r <= r)) * 100
        note = "top 1%" if pct >= 99 else ("top 5%" if pct >= 95 else "")
        print(f"  ({a:2d}, {b:2d})      {r:>8.4f}   {pct:>10.1f}th   {note}")

    print("\nTop-10 most correlated pairs (all classes):")
    print(f"  {'Pair':<12}  {'r':>8}")
    print("  " + "-" * 24)
    order = np.argsort(all_r)[::-1]
    for k in range(min(10, len(order))):
        idx = order[k]
        a, b = int(i_upper[idx]), int(j_upper[idx])
        marker = " ← confused pair" if (a, b) in CONFUSED_PAIRS or (b, a) in CONFUSED_PAIRS else ""
        print(f"  ({a:2d}, {b:2d})      {all_r[idx]:>8.4f}{marker}")
    print("=" * 55)


def softmax_entropy():
    """
    For each difficult class, compute Shannon entropy of the classifier's softmax
    output on every real sample in that class.

    Low entropy + wrong prediction  → the model has confidently associated this
        class's expression pattern with a different label (systematic misclassification).
    High entropy                    → genuine ambiguity; the pattern sits between
        multiple plausible classes and the model is uncertain.

    Maximum possible entropy = log(num_classes) ≈ 3.99 nats for 54 classes.

    Saves: evaluation/difficult_classes/softmax_entropy.png
    Prints: mean entropy, % of max, correct count, and top-3 predicted classes per
            difficult class.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y = _load_real_data()
    model = _load_classifier()
    num_classes = int(y.max()) + 1
    max_entropy = np.log(num_classes)

    print(f"\nMax entropy for {num_classes} classes: {max_entropy:.4f} nats\n")

    fig, axes = plt.subplots(1, len(DIFFICULT_CLASSES), figsize=(14, 4), sharey=False)

    for ax, cls in zip(axes, DIFFICULT_CLASSES):
        mask = y == cls
        samples = X[mask]
        if samples.ndim == 3:
            samples = samples[..., np.newaxis]

        probs = model.predict(samples, verbose=0)                       # (N, num_classes)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)       # (N,)
        mean_ent = float(entropy.mean())
        mean_ent_pct = mean_ent / max_entropy * 100

        pred_classes = np.argmax(probs, axis=1)
        correct = int(np.sum(pred_classes == cls))
        unique, counts = np.unique(pred_classes, return_counts=True)
        top3 = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:3]

        print(f"Class {cls:2d}: n={len(samples):3d}  "
              f"mean_entropy={mean_ent:.3f} ({mean_ent_pct:.1f}% of max)  "
              f"correct={correct}/{len(samples)}  top_predictions={top3}")

        ax.hist(entropy, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(max_entropy, color='red', linestyle='--', linewidth=1.2,
                   label=f'Max ({max_entropy:.2f})')
        ax.axvline(mean_ent, color='orange', linestyle='-', linewidth=1.5,
                   label=f'Mean ({mean_ent:.2f})')
        ax.set_title(f"Class {cls}  n={len(samples)}\ncorrect={correct}/{len(samples)}")
        ax.set_xlabel("Softmax entropy (nats)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Classifier Softmax Entropy — Difficult Classes\n"
        "(red dashed = max entropy / fully uncertain; orange = sample mean)",
        fontsize=11
    )
    out_path = os.path.join(OUTPUT_DIR, "softmax_entropy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def within_between_class_distance():
    """
    For each confused pair (a, b), compare:
      - Between-centroid distance: L2 distance between the two class mean images.
      - Within-class spread A/B: mean L2 distance of each sample from its class centroid.

    Ratio = between-centroid / mean(spread_a, spread_b).
    Ratio < 1 → the class distributions overlap in pixel space and are inherently
    confusable regardless of classifier capacity.

    Saves: evaluation/difficult_classes/within_between_distance.png
    Prints: table of distances and ratios for each pair.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y = _load_real_data()
    flat = X.reshape(len(X), -1).astype(np.float64)  # (N, H*W)

    print(f"\n{'Pair':>10}  {'Between-centroid':>18}  {'Spread A':>10}  "
          f"{'Spread B':>10}  {'Ratio':>8}")
    print("-" * 66)

    pair_labels, between_dists, spreads_a, spreads_b = [], [], [], []

    for a, b in CONFUSED_PAIRS:
        flat_a = flat[y == a]
        flat_b = flat[y == b]
        centroid_a = flat_a.mean(axis=0)
        centroid_b = flat_b.mean(axis=0)

        between  = float(np.linalg.norm(centroid_a - centroid_b))
        spread_a = float(np.linalg.norm(flat_a - centroid_a, axis=1).mean())
        spread_b = float(np.linalg.norm(flat_b - centroid_b, axis=1).mean())
        ratio    = between / ((spread_a + spread_b) / 2)

        print(f"  ({a:2d},{b:2d}):    {between:>14.2f}    {spread_a:>8.2f}    "
              f"{spread_b:>8.2f}   {ratio:>6.3f}")

        pair_labels.append(f"({a},{b})")
        between_dists.append(between)
        spreads_a.append(spread_a)
        spreads_b.append(spread_b)

    print("\nRatio < 1 → between-centroid distance < average within-class spread "
          "→ classes are inherently confusable.")

    x_pos = np.arange(len(CONFUSED_PAIRS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width, between_dists, width, label='Between-centroid distance', color='tomato')
    ax.bar(x_pos,         spreads_a,     width, label='Within-class spread (A)',   color='steelblue', alpha=0.85)
    ax.bar(x_pos + width, spreads_b,     width, label='Within-class spread (B)',   color='darkorange', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_labels)
    ax.set_xlabel("Confused pair (class A, class B)")
    ax.set_ylabel("Mean L2 distance (pixel space)")
    ax.set_title(
        "Within-class Spread vs Between-centroid Distance\n"
        "(red bar < blue/orange → class distributions overlap in pixel space)"
    )
    ax.legend()

    out_path = os.path.join(OUTPUT_DIR, "within_between_distance.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic analysis for difficult-to-classify brain sub-region classes"
    )
    parser.add_argument(
        '--mode',
        choices=['all', 'correlation', 'entropy', 'distance'],
        default='all',
        help=(
            '"correlation" — centroid Pearson r stats table for confused pairs; '
            '"entropy"     — classifier softmax entropy on real samples; '
            '"distance"    — within-class vs between-centroid L2 distance; '
            '"all"         — run all three (default)'
        )
    )
    args = parser.parse_args()

    if args.mode in ('all', 'correlation'):
        centroid_correlation_stats()
    if args.mode in ('all', 'entropy'):
        softmax_entropy()
    if args.mode in ('all', 'distance'):
        within_between_class_distance()
