#!/usr/bin/env python
"""
Encoding fidelity gate (M3.5).

Real genes -> images -> back to genes, scored with Viñas et al.'s own metrics.
Run before training any generative model, so that a later shortfall can be
attributed to the model rather than to the representation.

WHAT THIS IS NOT
    Their §5.2 numbers (S_dist 0.920) compare real test data against GAN output.
    A roundtrip compares real data against itself through a near-lossless codec:
    same samples, same order, no sampling noise, and ~99.9% of genes holding a
    dedicated pixel-channel slot. The expected S_dist is ~0.999. Scoring 0.920
    here would be an alarm, not a success. Their published bound (0.947) is a
    floor to clear comfortably, not a target.

WHAT IT IS FOR
    1. Catching silent bugs — gene_pixel_channel indexing, channel_scales
       mis-scaling, gene-order drift, padding off-by-one. These produce
       plausible-looking images and garbage genes, and nothing else flags them.
    2. Establishing the ceiling that M5 reads its synthetic scores against.
    3. Building the harness M5 reuses unchanged, with decoded synthetic data in
       place of roundtripped real data.

    At 16 channels (1) nearly cannot fail. Run it across 128x128x16, 256x256x1
    and 512x512x1 and it stops being pass/fail: at one channel every gene is
    collision-averaged, so the sweep measures fidelity against the channel budget
    on the reference paper's own metric.

USAGE
    RUN_DATASET=rnaseqdb python -m evaluation.roundtrip_fidelity
    RUN_DATASET=rnaseqdb python -m evaluation.roundtrip_fidelity --width 256 --height 256 --channels 1
    RUN_DATASET=rnaseqdb python -m evaluation.roundtrip_fidelity --max-genes 3000 --skip-tstr  # fast local
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from evaluation import vinas_metrics as vm
from preprocessing.artifact_paths import PreprocessingConfig
from preprocessing.gene_vector_reconstruction import (
    compute_exact_mask,
    reconstruct_gene_vectors,
)

# Published reference points (Wang-corpus rows, Viñas et al. Table 2)
VINAS_GAN_S_DIST = 0.920
VINAS_BOUND_S_DIST = 0.947
VINAS_GAN_S_DEND = 0.215
VINAS_BOUND_S_DEND = 0.222

GATE_MAX_LOSSY_FRAC = 0.002
GATE_MIN_MEDIAN_PEARSON = 0.999
GATE_MIN_S_DIST = 0.99
GATE_MAX_TSTR_DROP = 0.01


def roundtrip(config, batch=512, max_samples=None):
    """Decode every image back to a gene vector, in batches to bound memory."""
    images = np.load(config.resized_expressions_path, mmap_mode="r")
    gpc = np.load(config.gene_pixel_channel_path)
    scales = np.load(config.channel_scales_path)
    n = len(images) if max_samples is None else min(max_samples, len(images))

    out = np.empty((n, len(gpc)), dtype=np.float32)
    for i in range(0, n, batch):
        chunk = np.asarray(images[i:min(i + batch, n)], dtype=np.float32)
        out[i:i + len(chunk)] = reconstruct_gene_vectors(chunk, gpc, scales)
    return out, gpc, scales


def per_gene_pearson(a, b):
    """Correlation of each gene's column between two matrices, ignoring constants."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    az = a - a.mean(0); bz = b - b.mean(0)
    denom = np.sqrt((az ** 2).sum(0) * (bz ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (az * bz).sum(0) / denom
    r[denom == 0] = 1.0  # a gene constant in both round-trips perfectly
    return r


def main():
    p = argparse.ArgumentParser(description="Encoding fidelity gate")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--max-genes", type=int, default=None,
                   help="subsample genes for the gamma metrics (memory: cost is quadratic)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--skip-tstr", action="store_true")
    p.add_argument("--skip-bound", action="store_true")
    p.add_argument("--tstr-max-iter", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = PreprocessingConfig(args.width, args.height, args.channels,
                                 dataset=os.environ.get("RUN_DATASET") or "gtex")
    out_dir = config.evaluation_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print(f"ENCODING FIDELITY GATE — {config.dataset} — {config.tag}")
    print("=" * 74)

    real_full = np.load(config.data_path, mmap_mode="r")
    rt, gpc, scales = roundtrip(config, max_samples=args.max_samples)
    real = np.asarray(real_full[:len(rt)], dtype=np.float32)
    print(f"  real {real.shape}   roundtrip {rt.shape}")

    # ---- 1. gene accounting ------------------------------------------------
    exact = compute_exact_mask(gpc, config.channels)
    lossy_frac = float((~exact).mean())
    print(f"\n[1] Gene accounting")
    print(f"    exact slots : {exact.sum():,} / {len(exact):,} ({100*exact.mean():.3f}%)")
    print(f"    lossy       : {(~exact).sum():,} ({100*lossy_frac:.3f}%)")

    # ---- 2. per-gene fidelity ---------------------------------------------
    err = np.abs(rt - real)
    r = per_gene_pearson(real, rt)
    med_r = float(np.median(r))
    print(f"\n[2] Per-gene fidelity")
    print(f"    max |err| exact genes : {err[:, exact].max():.3e}")
    if (~exact).any():
        print(f"    max |err| lossy genes : {err[:, ~exact].max():.3e}")
    print(f"    Pearson median={med_r:.6f}  p1={np.percentile(r,1):.4f}  min={np.nanmin(r):.4f}")

    # ---- 3. gamma metrics --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    gene_idx = slice(None)
    if args.max_genes and args.max_genes < real.shape[1]:
        gene_idx = np.sort(rng.choice(real.shape[1], args.max_genes, replace=False))
        print(f"\n    (gamma on a {args.max_genes}-gene subsample)")

    # standardize with the REAL data's statistics, applied to both
    a = real[:, gene_idx].astype(np.float64)
    b = rt[:, gene_idx].astype(np.float64)
    mu, sd = a.mean(0), a.std(0) + 1e-8
    a_s, b_s = (a - mu) / sd, (b - mu) / sd

    print(f"\n[3] Gamma metrics (real vs roundtrip, {a_s.shape[1]} genes)")
    g = vm.gamma_scores(a_s, b_s)
    print(f"    S_dist = {g['s_dist']:.6f}      (Viñas GAN {VINAS_GAN_S_DIST}, bound {VINAS_BOUND_S_DIST})")
    print(f"    S_dend = {g['s_dend']:.6f}      (Viñas GAN {VINAS_GAN_S_DEND}, bound {VINAS_BOUND_S_DEND})")

    bound = None
    if not args.skip_bound:
        print(f"\n[4] Sampling bound (real vs real, disjoint halves, 5 runs)")
        bound = vm.sampling_bound(a_s, n_runs=5, seed=args.seed)
        print(f"    S_dist = {bound['mean']['s_dist']:.6f} ± {bound['std']['s_dist']:.6f}")
        print(f"    S_dend = {bound['mean']['s_dend']:.6f} ± {bound['std']['s_dend']:.6f}")
        print(f"    The roundtrip has no sampling noise, so it should sit ABOVE this.")

    # ---- 5. TSTR -----------------------------------------------------------
    tstr = {}
    if not args.skip_tstr:
        from sklearn.model_selection import train_test_split
        labels = pd.read_csv(config.labels_path).iloc[:len(rt)]
        print(f"\n[5] TSTR — train on roundtrip, test on real (MLP 64x64, 5 runs)")
        for attr in ("tissue", "condition"):
            if attr not in labels.columns:
                continue
            y = pd.factorize(labels[attr], sort=True)[0]
            tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                                      random_state=1, stratify=y)
            got = vm.tstr_scores(b_s[tr], y[tr], a_s[te], y[te],
                                 seed=args.seed, max_iter=args.tstr_max_iter)
            base = vm.tstr_scores(a_s[tr], y[tr], a_s[te], y[te],
                                  seed=args.seed, max_iter=args.tstr_max_iter)
            tstr[attr] = {"roundtrip": got, "real_baseline": base}
            print(f"    {attr:<10} roundtrip AUC={got['mean']['auc']:.4f} "
                  f"F1={got['mean']['f1_macro']:.4f}")
            print(f"    {'':<10} baseline  AUC={base['mean']['auc']:.4f} "
                  f"F1={base['mean']['f1_macro']:.4f}")

    # ---- gate --------------------------------------------------------------
    checks = [
        (f"lossy genes <= {GATE_MAX_LOSSY_FRAC:.1%}", lossy_frac <= GATE_MAX_LOSSY_FRAC,
         f"{100*lossy_frac:.3f}%"),
        (f"median Pearson >= {GATE_MIN_MEDIAN_PEARSON}", med_r >= GATE_MIN_MEDIAN_PEARSON,
         f"{med_r:.6f}"),
        (f"S_dist >= {GATE_MIN_S_DIST}", g["s_dist"] >= GATE_MIN_S_DIST, f"{g['s_dist']:.6f}"),
    ]
    if bound is not None:
        checks.append(("S_dist above sampling bound", g["s_dist"] > bound["mean"]["s_dist"],
                       f"{g['s_dist']:.6f} vs {bound['mean']['s_dist']:.6f}"))
    for attr, v in tstr.items():
        drop = v["real_baseline"]["mean"]["f1_macro"] - v["roundtrip"]["mean"]["f1_macro"]
        checks.append((f"TSTR {attr} within {GATE_MAX_TSTR_DROP} of baseline",
                       drop <= GATE_MAX_TSTR_DROP, f"drop={drop:+.4f}"))

    print("\n" + "=" * 74)
    print("M3.5 GATE")
    print("=" * 74)
    ok = True
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<38} {detail}")

    payload = {
        "config": {"dataset": config.dataset, "tag": config.tag,
                   "n_samples": int(len(rt)), "n_genes": int(real.shape[1]),
                   "gamma_genes": int(a_s.shape[1])},
        "gene_accounting": {"exact": int(exact.sum()), "lossy": int((~exact).sum()),
                            "lossy_frac": lossy_frac},
        "per_gene_pearson": {"median": med_r, "p1": float(np.percentile(r, 1)),
                             "min": float(np.nanmin(r))},
        "gamma": g, "sampling_bound": bound, "tstr": tstr,
        "reference": {"vinas_gan_s_dist": VINAS_GAN_S_DIST,
                      "vinas_bound_s_dist": VINAS_BOUND_S_DIST},
        "passed": bool(ok),
    }
    path = os.path.join(out_dir, f"roundtrip_fidelity_{config.tag}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Saved {path}")
    print("\n" + ("M3.5 GATE PASSED" if ok else "M3.5 GATE FAILED") + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
