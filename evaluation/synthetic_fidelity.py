#!/usr/bin/env python
"""
Synthetic-vs-real evaluation (M5).

Generates images from a trained diffusion model, decodes them back to gene
vectors, and scores them against held-out real data with Viñas et al.'s own
§5.2 metrics — the same harness M3.5 used, with synthetic data in place of
roundtripped real data.

WHAT CHANGES FROM M3.5
    M3.5 compared real against a roundtrip of ITSELF: paired, same samples, same
    order. Per-gene error and per-gene Pearson were meaningful there because row
    i on both sides is the same biological sample.

    Here the two sets are UNPAIRED — synthetic sample i corresponds to no real
    sample — so those two sections are gone. What survives is exactly the part of
    the harness that was always distributional: the gamma coefficients, which
    compare gene-gene correlation structure, and TSTR, which asks whether a
    classifier trained on synthetic data works on real data.

    M3.5's numbers are the CEILING for this run. The encoding lost 0.072% of
    genes to collisions before any model was involved; no generator reading
    through that encoding can beat it. Report M5 against that ceiling, not
    against 1.0.

WHY THE SYNTHETIC SET IS COMPOSITION-MATCHED
    Synthetic samples are conditioned on the label rows of the real test split,
    one for one, so both sides carry the same tissue/condition mix. Generating a
    flat N-per-class set instead would make the gamma coefficients partly measure
    a composition shift rather than sample quality. Conditioning on a test row's
    LABELS is not leakage — no expression value from the test split reaches the
    generator, and the model never trained on those samples.

WHY THE SPLIT COMES FROM classifer.training_data
    The same make_split() the classifiers call, so an M5 number and a classifier
    number refer to the same held-out samples. --split vinas reproduces the
    reference paper's procedure and is the default here for that reason.

USAGE
    RUN_DATASET=rnaseqdb python -m evaluation.synthetic_fidelity \
        --checkpoint output/diffusion/local/rnaseqdb/checkpoints/diffusion_model_ema.weights.h5

    # bounded local sanity check
    RUN_DATASET=rnaseqdb python -m evaluation.synthetic_fidelity --checkpoint ... \
        --max-samples 64 --max-genes 2000 --num-steps 4 --skip-bound
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from classifer.training_data import make_split
from diffusion.diffusion_config import get_config
from diffusion.diffusion_edm_sample import generate_conditioned_batch
from diffusion.diffusion_model import build_unet
from evaluation import vinas_metrics as vm
from preprocessing.artifact_paths import PreprocessingConfig
from preprocessing.gene_vector_reconstruction import reconstruct_gene_vectors

# Published reference points (Wang-corpus rows, Viñas et al. Table 2)
VINAS_GAN_S_DIST = 0.920
VINAS_GAN_S_DEND = 0.215
VINAS_BOUND_S_DIST = 0.947
VINAS_BOUND_S_DEND = 0.222


def load_label_codes(config, attribute_sizes):
    """[N, A] integer codes in the order the model's embedding tables expect."""
    codes = []
    for name, vocab_size in attribute_sizes:
        one_hot = np.load(config.y_attribute_path(name))
        if one_hot.shape[1] != vocab_size:
            raise ValueError(
                f"y_{name}.npy has {one_hot.shape[1]} columns, config declares {vocab_size}"
            )
        codes.append(one_hot.argmax(axis=1).astype(np.int32))
    return np.stack(codes, axis=1)


def decode(images, config, batch=256):
    """Images -> gene vectors, batched so peak memory stays near one chunk."""
    gpc = np.load(config.gene_pixel_channel_path)
    scales = np.load(config.channel_scales_path)
    out = np.empty((len(images), len(gpc)), dtype=np.float32)
    for i in range(0, len(images), batch):
        chunk = np.asarray(images[i:i + batch], dtype=np.float32)
        out[i:i + len(chunk)] = reconstruct_gene_vectors(chunk, gpc, scales)
    return out


def main():
    p = argparse.ArgumentParser(description="Synthetic-vs-real fidelity (M5)")
    p.add_argument("--checkpoint", required=True, help="trained .weights.h5 (EMA preferred)")
    p.add_argument("--mode", default="local", choices=("local", "diagnostic"),
                   help="which diffusion config the checkpoint was trained under")
    p.add_argument("--split", default="vinas", choices=("stratified", "vinas", "donor"))
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--num-steps", type=int, default=40, help="ODE denoising steps")
    p.add_argument("--batch-size", type=int, default=32, help="generation chunk size")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap the test set. Bounded local sanity checks only — generation "
                        "is the expensive step and scales linearly with this.")
    p.add_argument("--max-genes", type=int, default=None,
                   help="subsample genes for the gamma metrics (cost is quadratic)")
    p.add_argument("--skip-tstr", action="store_true")
    p.add_argument("--skip-bound", action="store_true")
    p.add_argument("--tstr-max-iter", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dcfg = get_config(args.mode)
    attribute_sizes = dcfg.get("attributes")
    config = PreprocessingConfig(dcfg["image_size"], dcfg["image_size"], dcfg["in_channels"],
                                 dataset=os.environ.get("RUN_DATASET") or "gtex")
    out_dir = config.evaluation_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print(f"SYNTHETIC FIDELITY (M5) — {config.dataset} — {config.tag}")
    print("=" * 74)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  guidance   : w={args.guidance_scale}   ODE steps={args.num_steps}")

    # ---- held-out split ----------------------------------------------------
    frame = pd.read_csv(config.labels_path)
    if attribute_sizes:
        codes = load_label_codes(config, attribute_sizes)
        strat = {"tissue": np.load(config.y_attribute_path("tissue")),
                 "condition": np.load(config.y_attribute_path("condition"))}
    else:
        one_hot = np.load(config.labels_path.replace("labels.csv", dcfg["label_file"]))
        codes = one_hot.argmax(axis=1).astype(np.int32)
        strat = {"tissue": one_hot}
    _, test_idx = make_split(strat, frame, mode=args.split)
    test_idx = np.sort(test_idx)
    if args.max_samples and args.max_samples < len(test_idx):
        # evenly spaced through the sorted test indices so the label mix survives
        pick = np.linspace(0, len(test_idx) - 1, args.max_samples).astype(np.int64)
        test_idx = test_idx[pick]
    print(f"  test split : {len(test_idx)} samples (mode={args.split})")

    # ---- real held-out genes ----------------------------------------------
    real = np.asarray(np.load(config.data_path, mmap_mode="r")[test_idx], dtype=np.float32)

    # ---- generate composition-matched synthetic images ---------------------
    print(f"\n[1] Generating {len(test_idx)} synthetic samples")
    model = build_unet(dcfg)
    model.load_weights(args.checkpoint)
    images = generate_conditioned_batch(
        model, dcfg, codes[test_idx],
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )

    print(f"\n[2] Decoding to gene vectors")
    synth = decode(images, config)
    del images
    print(f"    real {real.shape}   synthetic {synth.shape}")

    # ---- gamma metrics -----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    gene_idx = slice(None)
    if args.max_genes and args.max_genes < real.shape[1]:
        gene_idx = np.sort(rng.choice(real.shape[1], args.max_genes, replace=False))
        print(f"    (gamma on a {args.max_genes}-gene subsample)")

    # Standardize with the REAL data's statistics, applied to both — the same
    # convention M3.5 used, so the two runs' numbers are directly comparable.
    a = real[:, gene_idx].astype(np.float64)
    b = synth[:, gene_idx].astype(np.float64)
    mu, sd = a.mean(0), a.std(0) + 1e-8
    a_s, b_s = (a - mu) / sd, (b - mu) / sd

    print(f"\n[3] Gamma metrics (real test vs synthetic, {a_s.shape[1]} genes)")
    g = vm.gamma_scores(a_s, b_s)
    print(f"    S_dist = {g['s_dist']:.6f}      (Viñas GAN {VINAS_GAN_S_DIST})")
    print(f"    S_dend = {g['s_dend']:.6f}      (Viñas GAN {VINAS_GAN_S_DEND})")

    bound = None
    if not args.skip_bound:
        print(f"\n[4] Sampling bound (real vs real, disjoint halves, 5 runs)")
        bound = vm.sampling_bound(a_s, n_runs=5, seed=args.seed)
        print(f"    S_dist = {bound['mean']['s_dist']:.6f} ± {bound['std']['s_dist']:.6f}")
        print(f"    S_dend = {bound['mean']['s_dend']:.6f} ± {bound['std']['s_dend']:.6f}")
        print("    This is the CEILING — two disjoint halves of real data score this.")

    # ---- TSTR --------------------------------------------------------------
    tstr = {}
    if not args.skip_tstr:
        from sklearn.model_selection import train_test_split
        labels = frame.iloc[test_idx]
        print(f"\n[5] TSTR — train on synthetic, test on real (MLP 64x64, 5 runs)")
        for attr in ("tissue", "condition"):
            if attr not in labels.columns:
                continue
            y = pd.factorize(labels[attr], sort=True)[0]
            counts = np.bincount(y)
            n_classes = int((counts > 0).sum())
            # Two separate requirements, both of which a bounded --max-samples run
            # can break: every class needs 2 members to be splittable at all, and
            # sklearn additionally needs the test side to be at least n_classes
            # wide to place one of each. Skip rather than crash — a bounded run is
            # checking the plumbing, and TSTR is not the part being checked.
            if n_classes < 2 or counts[counts > 0].min() < 2:
                print(f"    {attr:<10} skipped — a class has <2 samples in this slice")
                continue
            if int(len(y) * 0.25) < n_classes:
                print(f"    {attr:<10} skipped — {len(y)} samples cannot give a "
                      f"stratified test side for {n_classes} classes")
                continue
            tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                                      random_state=1, stratify=y)
            got = vm.tstr_scores(b_s[tr], y[tr], a_s[te], y[te],
                                 seed=args.seed, max_iter=args.tstr_max_iter)
            base = vm.tstr_scores(a_s[tr], y[tr], a_s[te], y[te],
                                  seed=args.seed, max_iter=args.tstr_max_iter)
            tstr[attr] = {"synthetic": got, "real_baseline": base}
            print(f"    {attr:<10} synthetic AUC={got['mean']['auc']:.4f} "
                  f"F1={got['mean']['f1_macro']:.4f}")
            print(f"    {'':<10} baseline  AUC={base['mean']['auc']:.4f} "
                  f"F1={base['mean']['f1_macro']:.4f}")

    # ---- report ------------------------------------------------------------
    ceiling = None
    ceiling_path = os.path.join(out_dir, f"roundtrip_fidelity_{config.tag}.json")
    if os.path.exists(ceiling_path):
        with open(ceiling_path) as fh:
            ceiling = json.load(fh).get("gamma")

    print("\n" + "=" * 74)
    print("M5 SUMMARY")
    print("=" * 74)
    print(f"  {'':<26}{'S_dist':>10}{'S_dend':>10}")
    print(f"  {'this model':<26}{g['s_dist']:>10.4f}{g['s_dend']:>10.4f}")
    if ceiling:
        print(f"  {'M3.5 encoding ceiling':<26}{ceiling['s_dist']:>10.4f}{ceiling['s_dend']:>10.4f}")
    if bound:
        print(f"  {'real-vs-real bound':<26}"
              f"{bound['mean']['s_dist']:>10.4f}{bound['mean']['s_dend']:>10.4f}")
    print(f"  {'Vinas et al. GAN':<26}{VINAS_GAN_S_DIST:>10.4f}{VINAS_GAN_S_DEND:>10.4f}")
    print(f"  {'Vinas et al. bound':<26}{VINAS_BOUND_S_DIST:>10.4f}{VINAS_BOUND_S_DEND:>10.4f}")

    payload = {
        "config": {"dataset": config.dataset, "tag": config.tag, "mode": args.mode,
                   "checkpoint": args.checkpoint, "split": args.split,
                   "guidance_scale": args.guidance_scale, "ode_steps": args.num_steps,
                   "n_samples": int(len(test_idx)), "n_genes": int(real.shape[1]),
                   "gamma_genes": int(a_s.shape[1])},
        "gamma": g, "sampling_bound": bound, "tstr": tstr,
        "encoding_ceiling": ceiling,
        "reference": {"vinas_gan_s_dist": VINAS_GAN_S_DIST,
                      "vinas_gan_s_dend": VINAS_GAN_S_DEND,
                      "vinas_bound_s_dist": VINAS_BOUND_S_DIST,
                      "vinas_bound_s_dend": VINAS_BOUND_S_DEND},
    }
    path = os.path.join(out_dir, f"synthetic_fidelity_{config.tag}_w{args.guidance_scale:.1f}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Saved {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
