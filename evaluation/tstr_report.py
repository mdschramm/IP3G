#!/usr/bin/env python
"""
M6 — the §5.2.2 comparison table.

Collects the four ClassiferSmall runs (tissue/condition x real/synthetic),
prints them beside Viñas et al.'s published numbers, and writes one JSON.

WHAT THIS TABLE DOES AND DOES NOT SHOW
    It compares two generative pipelines on the same protocol, not two numbers on
    the same data. Their classifier is an MLP over 18,154-dim expression vectors;
    ours is a CNN over 128x128x16 expression images. The comparison is meaningful
    because the PROTOCOL is held fixed — same 75/25 procedure, same
    composition-matched synthetic training set, same held-out real test set,
    same averaging over 5 runs — and because the real-trained row is carried in
    both columns as the within-pipeline reference.

    Read the synthetic row against ITS OWN real row first. The gap between them
    is what the generator costs; the gap between our real row and theirs is a
    difference of classifier and representation, and confounds it.

THREE STANDING CAVEATS, PRINTED ON EVERY RUN
    donor leakage  The vinas split does not group by donor, so the same
                   individuals appear on both sides. This inflates every row
                   here, ours and theirs — it is inherent to reproducing their
                   procedure, which used the plain split_train_test(), not the
                   patient-aware split_train_test_v2() in the same utils file.
    encoding cost  0.072% of genes are lost to pixel collisions before any model
                   is involved (M3.5). The roundtrip TSTR row is the real ceiling
                   for the synthetic row, not 1.0.
    sigma_data     The EDM preconditioning constant was measured over the full
                   corpus during preprocessing, so it carries a whiff of test
                   data. One global scalar, unrelated to labels.

USAGE
    RUN_DATASET=rnaseqdb python -m evaluation.tstr_report
"""

import argparse
import glob
import json
import os

import numpy as np

from preprocessing.artifact_paths import DEFAULT_CONFIG, PreprocessingConfig, model_output_dir

# Viñas et al. 2022, §5.2.2 (Wang corpus). AUC, F1 as mean +/- std over 5 runs.
VINAS = {
    ("tissue", "synthetic"): (0.9884, 0.0010, 0.9222, 0.0040),
    ("tissue", "real"): (0.9986, 0.0003, 0.9860, 0.0007),
    ("condition", "synthetic"): (0.9992, 0.0001, 0.9893, 0.0009),
    ("condition", "real"): (0.9997, 0.0001, 0.9939, 0.0005),
}

ROWS = [("tissue", "real"), ("tissue", "synthetic"),
        ("condition", "real"), ("condition", "synthetic")]


def load_metrics(metrics_dir, attribute, kind):
    """Find the run for one (attribute, kind) cell, by suffix then by content.

    The suffix convention (`--out-suffix real_tissue`) is the documented path;
    the content fallback exists because a mislabelled suffix would otherwise put
    a synthetic-trained run in the real-trained row, which is precisely the error
    this table cannot afford to make silently.
    """
    preferred = os.path.join(metrics_dir, f"classifier_small_metrics_{kind}_{attribute}.json")
    candidates = [preferred] if os.path.exists(preferred) else sorted(
        glob.glob(os.path.join(metrics_dir, "classifier_small_metrics*.json")))
    for path in candidates:
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            continue
        got_kind = "synthetic" if str(data.get("trained_on", "")).startswith("synthetic") else "real"
        if data.get("attribute") == attribute and got_kind == kind:
            return path, data
    return None, None


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"


def main():
    p = argparse.ArgumentParser(description="M6 TSTR comparison table")
    p.add_argument("--metrics-dir", default=None,
                   help="where the classifier_small_metrics_*.json live "
                        "(default: the classifier output dir for this RUN_MODE/RUN_DATASET)")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--channels", type=int, default=16)
    args = p.parse_args()

    config = PreprocessingConfig(args.width, args.height, args.channels,
                                 dataset=DEFAULT_CONFIG.dataset)
    metrics_dir = args.metrics_dir or model_output_dir("classifier")

    print("=" * 86)
    print(f"M6 — TRAIN ON SYNTHETIC, TEST ON REAL — {config.dataset} — {config.tag}")
    print("=" * 86)
    print(f"  metrics: {metrics_dir}\n")

    header = f"  {'row':<30} {'ours AUC':<18} {'ours F1 (macro)':<18} {'Viñas AUC':<10} {'Viñas F1':<10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    collected, missing = {}, []
    for attribute, kind in ROWS:
        path, data = load_metrics(metrics_dir, attribute, kind)
        v_auc, v_auc_s, v_f1, v_f1_s = VINAS[(attribute, kind)]
        label = f"{attribute} / {kind}-trained"
        if data is None:
            missing.append(label)
            print(f"  {label:<30} {'—':<18} {'—':<18} {v_auc:<10.4f} {v_f1:<10.4f}")
            continue
        mean, std = data["mean"], data["std"]
        collected[f"{attribute}_{kind}"] = {
            "source": path, "n_runs": data.get("n_runs"),
            "n_train": data.get("n_train"), "n_val": data.get("n_val"),
            "ours": {"auc": mean["auc"], "auc_std": std["auc"],
                     "f1_macro": mean["f1_macro"], "f1_macro_std": std["f1_macro"],
                     "f1_weighted": mean["f1_weighted"], "accuracy": mean["accuracy"]},
            "vinas": {"auc": v_auc, "auc_std": v_auc_s, "f1": v_f1, "f1_std": v_f1_s},
        }
        print(f"  {label:<30} {fmt(mean['auc'], std['auc']):<18} "
              f"{fmt(mean['f1_macro'], std['f1_macro']):<18} {v_auc:<10.4f} {v_f1:<10.4f}")

    # The cost of the generator, within our own pipeline. This is the number the
    # table exists to produce; the cross-pipeline columns are context for it.
    print()
    for attribute in ("tissue", "condition"):
        real = collected.get(f"{attribute}_real")
        syn = collected.get(f"{attribute}_synthetic")
        if real and syn:
            d_auc = syn["ours"]["auc"] - real["ours"]["auc"]
            d_f1 = syn["ours"]["f1_macro"] - real["ours"]["f1_macro"]
            v_d_auc = VINAS[(attribute, "synthetic")][0] - VINAS[(attribute, "real")][0]
            v_d_f1 = VINAS[(attribute, "synthetic")][2] - VINAS[(attribute, "real")][2]
            print(f"  {attribute:<10} synthetic - real:  ours ΔAUC={d_auc:+.4f} ΔF1={d_f1:+.4f}"
                  f"    |  theirs ΔAUC={v_d_auc:+.4f} ΔF1={v_d_f1:+.4f}")

    ceiling = read_ceiling(config)
    if ceiling:
        print("\n  Ceilings (M3.5, encoding only — no generator involved):")
        for attribute, row in ceiling.items():
            print(f"    {attribute:<10} roundtrip AUC={row['roundtrip']['auc']:.4f} "
                  f"F1={row['roundtrip']['f1_macro']:.4f}   "
                  f"baseline AUC={row['baseline']['auc']:.4f} F1={row['baseline']['f1_macro']:.4f}")

    print("\n  Caveats — these apply to every row above:")
    print("    · donor leakage: the vinas split does not group by donor; 571 donors appear")
    print("      on both sides. Inherent to the reference procedure, inflates all rows.")
    print("    · encoding: 0.072% of genes lost to pixel collisions before any model (M3.5).")
    print("    · sigma_data was measured over the full corpus, not the train split alone.")
    print("    · their classifier is an MLP over gene vectors; ours is a CNN over images.")

    out_dir = config.evaluation_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tstr_report_{config.tag}.json")
    with open(out_path, "w") as fh:
        json.dump({"dataset": config.dataset, "tag": config.tag,
                   "rows": collected, "missing": missing,
                   "vinas_reference": {f"{a}_{k}": dict(zip(("auc", "auc_std", "f1", "f1_std"), v))
                                       for (a, k), v in VINAS.items()},
                   "encoding_ceiling": ceiling,
                   "caveats": {"shared_donors": 571,
                               "lossy_gene_fraction": 0.00072,
                               "sigma_data_scope": "full corpus"}}, fh, indent=2)
    print(f"\n  saved {out_path}")
    if missing:
        print(f"  MISSING {len(missing)} of 4 rows: {', '.join(missing)}")


def read_ceiling(config):
    """The M3.5 roundtrip TSTR rows, if that gate has been run for this corpus."""
    path = os.path.join(config.evaluation_dir, f"roundtrip_fidelity_{config.tag}.json")
    if not os.path.exists(path):
        return None
    try:
        data = json.load(open(path))
    except (OSError, ValueError):
        return None
    tstr = data.get("tstr")
    if not tstr:
        return None
    out = {}
    for attribute, row in tstr.items():
        rt, base = row.get("roundtrip", {}), row.get("real_baseline", {})
        if "mean" in rt and "mean" in base:
            out[attribute] = {"roundtrip": rt["mean"], "baseline": base["mean"]}
    return out or None


if __name__ == "__main__":
    main()
