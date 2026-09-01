"""M6b — TSTR on DECODED gene vectors, the row that lines up with Viñas §5.2.2.

M6 trains a CNN on synthetic *images*. The reference paper trains an MLP on
synthetic *gene vectors*. Those two numbers differ in the generator AND in the
classifier family AND in the data representation, so a gap between them cannot be
attributed to anything in particular.

This script removes two of those three. It takes the same synthetic replica, runs
it back through the inverse gene->pixel->channel map, and hands the resulting
[N, 18154] vectors to the same MLP(64, 64) harness the paper describes. What is
left differing from Viñas is the generator — which is the comparison we actually
want to make.

Three properties keep it honest:

  * The training rows are the decoded replica of the 6,860-sample TRAIN split.
    The test rows are the 2,287 real held-out samples. Same partition the
    diffusion model was trained under, verified against split_indices.npz.
    M5's TSTR does none of this — it splits a 512-sample slice internally and
    never touches the held-out set, so it is a fidelity probe, not this.

  * Standardisation uses REAL TRAIN statistics only. M3.5 and M5 standardise with
    the real statistics of whatever slice they hold, which for them is the only
    real data in scope. Here real train exists and is the leak-free reference, so
    it is what gets used; the difference is numerically tiny but it is the
    defensible choice and it is recorded in the output JSON.

  * The ceiling is printed alongside. M3.5 ran this same harness with the
    generator removed — encode/decode only. A synthetic score cannot exceed it,
    and reading M6b against 1.0 instead of against that row overstates the gap.

Usage:
    RUN_DATASET=rnaseqdb python -m evaluation.decoded_tstr \
        --synthetic-dir output/diffusion/diagnostic/rnaseqdb/synthetic_w1 \
        --mode diagnostic

    # bounded local probe
    RUN_DATASET=rnaseqdb python -m evaluation.decoded_tstr \
        --synthetic-dir <probe replica> --mode local \
        --max-samples 32 --tstr-max-iter 20 --n-runs 1
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from diffusion.diffusion_config import get_config
from diffusion.generate_synthetic_replica import resolve_split
from evaluation import vinas_metrics as vm
from preprocessing.artifact_paths import PreprocessingConfig
from preprocessing.gene_vector_reconstruction import reconstruct_gene_vectors

# Viñas et al. 2022, §5.2.2 — MLP over gene vectors, 2,287 held-out real samples.
VINAS = {
    "tissue":    {"synthetic": (0.9884, 0.9222), "real": (0.9986, 0.9860)},
    "condition": {"synthetic": (0.9992, 0.9893), "real": (0.9997, 0.9939)},
}


def decode(images, config, batch=256):
    """Images -> gene vectors, batched so peak memory stays near one chunk.

    Same body as synthetic_fidelity.decode; duplicated rather than imported
    because importing that module pulls in build_unet and the whole TF stack,
    and nothing here needs a GPU.
    """
    gpc = np.load(config.gene_pixel_channel_path)
    scales = np.load(config.channel_scales_path)
    out = np.empty((len(images), len(gpc)), dtype=np.float32)
    for i in range(0, len(images), batch):
        chunk = np.asarray(images[i:i + batch], dtype=np.float32)
        out[i:i + len(chunk)] = reconstruct_gene_vectors(chunk, gpc, scales)
    return out


def distribution_report(synth, real_train):
    """Do the decoded values look like expression at all?

    A generator can score well on TSTR while producing vectors that are obviously
    not expression data — a classifier only needs the classes to stay separable.
    These are the checks that catch that: per-gene first and second moments, and
    how far outside the real per-gene range the synthetic values stray.
    """
    s_mean, r_mean = synth.mean(0), real_train.mean(0)
    s_std, r_std = synth.std(0), real_train.std(0)
    lo, hi = real_train.min(0), real_train.max(0)

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 0 else float("nan")

    return {
        "per_gene_mean_pearson": pearson(s_mean, r_mean),
        "per_gene_std_pearson": pearson(s_std, r_std),
        "mean_abs_gene_mean_shift": float(np.abs(s_mean - r_mean).mean()),
        "mean_abs_gene_std_shift": float(np.abs(s_std - r_std).mean()),
        "frac_below_real_gene_min": float((synth < lo).mean()),
        "frac_above_real_gene_max": float((synth > hi).mean()),
        "frac_nonfinite": float((~np.isfinite(synth)).mean()),
        "frac_exact_zero": {"synthetic": float((synth == 0).mean()),
                            "real": float((real_train == 0).mean())},
        "global": {
            "synthetic": {"min": float(synth.min()), "max": float(synth.max()),
                          "mean": float(synth.mean()), "std": float(synth.std())},
            "real": {"min": float(real_train.min()), "max": float(real_train.max()),
                     "mean": float(real_train.mean()), "std": float(real_train.std())},
        },
        "row_sum": {"synthetic": float(synth.sum(1).mean()),
                    "real": float(real_train.sum(1).mean())},
    }


def load_ceiling(out_dir, tag):
    """M3.5's TSTR rows, if the gate has been run for this config."""
    path = os.path.join(out_dir, f"roundtrip_fidelity_{tag}.json")
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path)).get("tstr") or None
    except (json.JSONDecodeError, OSError):
        return None


def main():
    p = argparse.ArgumentParser(description="TSTR on decoded gene vectors (M6b)")
    p.add_argument("--synthetic-dir", required=True,
                   help="replica directory written by generate_synthetic_replica.py")
    p.add_argument("--mode", default="diagnostic", choices=("local", "diagnostic"),
                   help="which diffusion config the replica was generated under")
    p.add_argument("--split", default="vinas", choices=("stratified", "vinas", "donor"))
    p.add_argument("--attributes", default="tissue,condition")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap BOTH sides. Bounded probes only — the reported run uses all rows.")
    p.add_argument("--max-genes", type=int, default=None,
                   help="subsample genes; probes only")
    p.add_argument("--n-runs", type=int, default=5, help="MLP restarts to average, paper uses 5")
    p.add_argument("--tstr-max-iter", type=int, default=300)
    p.add_argument("--skip-diagnostics", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dcfg = get_config(args.mode)
    config = PreprocessingConfig(dcfg["image_size"], dcfg["image_size"], dcfg["in_channels"],
                                 dataset=os.environ.get("RUN_DATASET") or "gtex")
    out_dir = config.evaluation_dir
    os.makedirs(out_dir, exist_ok=True)
    attrs = [a for a in args.attributes.split(",") if a]

    print("=" * 74)
    print(f"DECODED-VECTOR TSTR (M6b) — {config.dataset} — {config.tag}")
    print("=" * 74)
    print(f"  replica    : {args.synthetic_dir}")

    # ---- split, verified against what the trainer actually used -------------
    frame, train_idx, test_idx = resolve_split(
        config, config, args.split, dcfg["checkpoint_dir"])

    syn_path = os.path.join(args.synthetic_dir, "resized_expressions.npy")
    syn_images = np.load(syn_path, mmap_mode="r")

    # A replica shorter than the train split is a bounded probe, not the real
    # thing. Fail loudly on a mismatch unless --max-samples says a probe is
    # intended: silently training on a truncated replica produces a plausible,
    # wrong number.
    if len(syn_images) != len(train_idx) and args.max_samples is None:
        raise ValueError(
            f"replica has {len(syn_images)} rows but the '{args.split}' train split has "
            f"{len(train_idx)}. Pass --max-samples to run this as a bounded probe."
        )

    # Which corpus row does each replica row correspond to? Resolve it by
    # sample_id from the replica's own labels.csv rather than by position. A
    # bounded probe replica holds evenly-spaced rows, not a head slice, so a
    # positional assumption is wrong there and would be invisible: the vectors
    # would still be correctly shaped, just paired with the wrong labels.
    syn_frame = pd.read_csv(os.path.join(args.synthetic_dir, "labels.csv"))
    if len(syn_frame) != len(syn_images):
        raise ValueError(
            f"{args.synthetic_dir}: labels.csv has {len(syn_frame)} rows but "
            f"resized_expressions.npy has {len(syn_images)}."
        )
    row_of = {s: i for i, s in enumerate(frame["sample_id"].to_numpy())}
    missing = [s for s in syn_frame["sample_id"] if s not in row_of]
    if missing:
        raise ValueError(
            f"{len(missing)} replica sample_ids are absent from {config.labels_path} "
            f"(first: {missing[0]}). The replica was built against a different corpus."
        )
    take_train = np.array([row_of[s] for s in syn_frame["sample_id"]], dtype=np.int64)

    stray = np.setdiff1d(take_train, train_idx)
    if len(stray):
        raise ValueError(
            f"{len(stray)} replica rows are NOT in the '{args.split}' train split. The "
            "classifier would be trained on synthesized copies of held-out samples and "
            "the TSTR number would be meaningless."
        )
    take_test = np.sort(test_idx)

    syn_pos = np.arange(len(syn_images))
    if args.max_samples:
        n = min(args.max_samples, len(take_train))
        pick = np.linspace(0, len(take_train) - 1, n).astype(np.int64)
        take_train, syn_pos = take_train[pick], syn_pos[pick]
        m = min(args.max_samples, len(take_test))
        take_test = take_test[np.linspace(0, len(take_test) - 1, m).astype(np.int64)]

    for attr in attrs:
        if attr in frame.columns and attr in syn_frame.columns:
            mine = frame[attr].to_numpy()[take_train]
            theirs = syn_frame[attr].to_numpy()[syn_pos]
            if not np.array_equal(mine, theirs):
                bad = int((mine != theirs).sum())
                raise ValueError(
                    f"replica row alignment is wrong: {bad}/{len(mine)} '{attr}' labels "
                    f"disagree after resolving by sample_id."
                )
    print("  alignment  : resolved by sample_id, labels verified, all rows in-split")
    print(f"  train rows : {len(take_train)} synthetic (decoded)")
    print(f"  test rows  : {len(take_test)} real, held out")

    # ---- decode; real vectors need no decoding, data.npy IS the gene matrix --
    print("\n[1] Decoding the replica through the inverse map")
    synth = decode(syn_images[syn_pos] if args.max_samples else syn_images, config)
    real_full = np.load(config.data_path, mmap_mode="r")
    real_train = np.asarray(real_full[take_train], dtype=np.float32)
    real_test = np.asarray(real_full[take_test], dtype=np.float32)
    print(f"    synthetic {synth.shape}   real train {real_train.shape}   "
          f"real test {real_test.shape}")

    gene_idx = slice(None)
    if args.max_genes and args.max_genes < synth.shape[1]:
        rng = np.random.default_rng(args.seed)
        gene_idx = np.sort(rng.choice(synth.shape[1], args.max_genes, replace=False))
        synth, real_train, real_test = (synth[:, gene_idx], real_train[:, gene_idx],
                                        real_test[:, gene_idx])
        print(f"    (gene subsample: {args.max_genes})")

    # ---- distribution diagnostics -------------------------------------------
    dist = None
    if not args.skip_diagnostics:
        print("\n[2] Do the decoded values look like expression?")
        dist = distribution_report(synth, real_train)
        print(f"    per-gene mean  Pearson : {dist['per_gene_mean_pearson']:.6f}")
        print(f"    per-gene std   Pearson : {dist['per_gene_std_pearson']:.6f}")
        print(f"    below real gene min    : {100*dist['frac_below_real_gene_min']:.3f}%")
        print(f"    above real gene max    : {100*dist['frac_above_real_gene_max']:.3f}%")
        print(f"    non-finite             : {100*dist['frac_nonfinite']:.3f}%")
        print(f"    mean row sum  synth {dist['row_sum']['synthetic']:.1f}  "
              f"real {dist['row_sum']['real']:.1f}")

    # ---- standardise on REAL TRAIN statistics, applied to all three ---------
    mu, sd = real_train.mean(0), real_train.std(0) + 1e-8
    synth_s = (synth - mu) / sd
    train_s = (real_train - mu) / sd
    test_s = (real_test - mu) / sd
    del synth, real_train, real_test

    # ---- TSTR ---------------------------------------------------------------
    print(f"\n[3] TSTR — MLP(64, 64) ReLU, {args.n_runs} runs, test on real held-out")
    tstr = {}
    for attr in attrs:
        if attr not in frame.columns:
            print(f"    {attr}: not in labels.csv, skipped")
            continue
        # Factorize over the FULL frame so train and test share one code space —
        # factorizing each side separately would silently relabel classes.
        codes = pd.factorize(frame[attr], sort=True)[0]
        y_tr, y_te = codes[take_train], codes[take_test]
        got = vm.tstr_scores(synth_s, y_tr, test_s, y_te, n_runs=args.n_runs,
                             seed=args.seed, max_iter=args.tstr_max_iter)
        base = vm.tstr_scores(train_s, y_tr, test_s, y_te, n_runs=args.n_runs,
                              seed=args.seed, max_iter=args.tstr_max_iter)
        tstr[attr] = {"synthetic": got, "real_baseline": base,
                      "n_classes": int(len(np.unique(np.concatenate([y_tr, y_te]))))}
        print(f"    {attr:<10} synthetic AUC={got['mean']['auc']:.4f} "
              f"F1={got['mean']['f1_macro']:.4f}")
        print(f"    {'':<10} real      AUC={base['mean']['auc']:.4f} "
              f"F1={base['mean']['f1_macro']:.4f}")

    # ---- the comparison table ----------------------------------------------
    ceiling = load_ceiling(out_dir, config.tag)
    print("\n" + "=" * 74)
    print("§5.2.2 — decoded gene vectors, MLP(64,64), tested on real held-out")
    print("=" * 74)
    print(f"{'task / trained on':<28}{'AUC':>10}{'F1 macro':>12}   {'Viñas AUC / F1':>20}")
    print("-" * 74)
    for attr, v in tstr.items():
        for row, key in (("synthetic", "synthetic"), ("real", "real_baseline")):
            m = v[key]["mean"]
            ref = VINAS.get(attr, {}).get(row)
            ref_s = f"{ref[0]:.4f} / {ref[1]:.4f}" if ref else "—"
            print(f"{attr + ' / ' + row:<28}{m['auc']:>10.4f}{m['f1_macro']:>12.4f}   {ref_s:>20}")
        if ceiling and attr in ceiling:
            c = ceiling[attr]["roundtrip"]["mean"]
            print(f"{'  ceiling (M3.5 encode only)':<28}{c['auc']:>10.4f}{c['f1_macro']:>12.4f}"
                  f"   {'—':>20}")
    print("-" * 74)
    print("Caveats on every row: the 'vinas' split does not group by donor; the encoding")
    print("loses 0.072% of genes before the model exists (M3.5 row is the ceiling); and")
    print("sigma_data was measured over the full corpus during preprocessing.")

    payload = {
        "dataset": config.dataset, "tag": config.tag, "split": args.split,
        "synthetic_dir": args.synthetic_dir, "mode": args.mode,
        "n_train": int(len(take_train)), "n_test": int(len(take_test)),
        "n_genes": int(len(mu)), "n_runs": args.n_runs,
        "tstr_max_iter": args.tstr_max_iter,
        "standardization": "real train mean/std, applied to synthetic and real test",
        "tstr": tstr, "distribution": dist,
        "ceiling_m35": ceiling, "vinas_reference": VINAS,
    }
    path = os.path.join(out_dir, f"decoded_tstr_{config.tag}.json")
    json.dump(payload, open(path, "w"), indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
