#!/usr/bin/env python
"""
Generate a synthetic replica of the diffusion model's own training split (M6).

WHY A REPLICA AND NOT M5's SYNTHETIC SET
    evaluation/synthetic_fidelity.py (M5) conditions on the TEST split's label
    rows, because it is comparing distributions against the test set. This script
    conditions on the TRAIN split's label rows, because it is building a drop-in
    replacement for the training data — Viñas et al. §5.2.2 trains a classifier
    on a synthetic set matched to the real training set in size and label
    composition, then scores it on the held-out real test set. Same generator,
    same guidance, different target; using M5's output here would train the
    classifier on the composition of the test set, which is the leak the whole
    protocol exists to avoid.

WHY IT IS WRITTEN THROUGH A MEMMAP
    6,860 x 128 x 128 x 16 float32 is 7.2 GB. generate_conditioned_batch returns
    its whole result in RAM, so it is called on slices and each slice is written
    straight into an np.lib.format.open_memmap array. Peak RSS stays at roughly
    one chunk plus the model. A progress.json cursor makes an interrupted run
    resume at the chunk boundary rather than regenerating hours of samples.

WHY THE SPLIT IS RECOMPUTED AND THEN CROSS-CHECKED
    make_split is deterministic, so recomputing it here is safe. It is checked
    against the split_indices.npz the trainer wrote anyway, because the one way
    this comparison can be silently wrong is a replica built against a different
    partition than the model was trained on: the classifier would then be trained
    on synthesized versions of the test set and report an excellent, meaningless
    number.

USAGE
    RUN_DATASET=rnaseqdb python -m diffusion.generate_synthetic_replica \
        --checkpoint output/diffusion/diagnostic/rnaseqdb/checkpoints/diffusion_model_ema.weights.h5 \
        --mode diagnostic --guidance-scale 2.0
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from classifer.training_data import describe_split, make_split
from diffusion.diffusion_config import get_config
from preprocessing.artifact_paths import DEFAULT_CONFIG, PreprocessingConfig
from preprocessing.label_frame import load_attribute_codes


def resolve_split(config, pcfg, mode, checkpoint_dir):
    """Recompute the partition and verify it against the trainer's record."""
    frame = pd.read_csv(pcfg.labels_path)
    strat = {a: np.load(pcfg.y_attribute_path(a)) for a in ("tissue", "condition")
             if os.path.exists(pcfg.y_attribute_path(a))}
    train_idx, test_idx = make_split(strat or None, frame, mode=mode)
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    record = os.path.join(checkpoint_dir, "split_indices.npz")
    if os.path.exists(record):
        saved = np.load(record, allow_pickle=True)
        if str(saved["mode"]) != mode:
            raise ValueError(
                f"{record} was written for split '{saved['mode']}' but this run asked for "
                f"'{mode}'. The replica would not match what the model trained on."
            )
        if not np.array_equal(np.sort(saved["train_idx"]), train_idx):
            raise ValueError(
                f"Recomputed '{mode}' split does not match {record}. The corpus or the "
                "split code changed since training; the replica would be built against a "
                "different partition than the model saw."
            )
        print(f"  Split verified against {record}")
    else:
        print(f"  No split_indices.npz in {checkpoint_dir} — split recomputed, unverified.")

    print(describe_split(frame, train_idx, test_idx, mode))
    return frame, train_idx, test_idx


def write_sidecars(out_dir, frame, attribute_sizes, pcfg, take):
    """labels.csv + one y_<attr>.npy per attribute, sliced to the generated rows.

    The point is that the output directory has the same shape as a preprocessing
    artifact directory, so ClassiferSmall can be pointed at it with --synthetic-dir
    and needs no special-casing for synthetic input.
    """
    frame.iloc[take].to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    for name, _ in attribute_sizes:
        one_hot = np.load(pcfg.y_attribute_path(name))[take]
        np.save(os.path.join(out_dir, f"y_{name}.npy"), one_hot)
    print(f"  Wrote labels.csv and {len(attribute_sizes)} y_<attr>.npy for {len(take)} rows")


def main():
    p = argparse.ArgumentParser(description="Synthetic replica of the training split (M6)")
    p.add_argument("--checkpoint", required=True, help="trained .weights.h5 (EMA preferred)")
    p.add_argument("--mode", default="diagnostic", choices=("local", "diagnostic"),
                   help="which diffusion config the checkpoint was trained under")
    p.add_argument("--split", default="vinas", choices=("stratified", "vinas", "donor"),
                   help="must match the --split the model was trained with")
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--num-steps", type=int, default=40, help="ODE steps per sample")
    p.add_argument("--batch-size", type=int, default=32, help="samples denoised at once")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="rows generated before flushing to disk; sets peak RAM and "
                        "resume granularity")
    p.add_argument("--output-dir", default=None,
                   help="default: <checkpoint_dir>/../synthetic_w<guidance>")
    p.add_argument("--max-samples", type=int, default=None,
                   help="bounded probe: evenly spaced within the train split")
    p.add_argument("--restart", action="store_true",
                   help="ignore any progress.json and regenerate from row 0")
    args = p.parse_args()

    config = get_config(args.mode)
    attribute_sizes = config.get("attributes")
    if not attribute_sizes:
        raise SystemExit(
            "This script targets the factorized conditioning path. Run it with "
            "RUN_DATASET=rnaseqdb (the flat GTEx path has no per-attribute labels)."
        )
    pcfg = PreprocessingConfig(config["image_size"], config["image_size"],
                               config["in_channels"], dataset=DEFAULT_CONFIG.dataset)

    print("=" * 74)
    print(f"SYNTHETIC REPLICA — {pcfg.dataset} — {pcfg.tag} — w={args.guidance_scale}")
    print("=" * 74)

    # The split record is read from the checkpoint's OWN directory, not from
    # config['checkpoint_dir']: --scratch-dir runs and hand-placed checkpoints both
    # put the weights somewhere the config does not point, and the split that
    # matters is the one written next to these weights.
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    frame, train_idx, _ = resolve_split(config, pcfg, args.split, ckpt_dir)
    take = train_idx
    if args.max_samples and args.max_samples < len(take):
        # Evenly spaced, not a head slice: labels.csv is grouped by tissue, so the
        # first N rows would be one tissue and a probe would never exercise the
        # conditioning tables it is meant to check.
        pick = np.linspace(0, len(take) - 1, args.max_samples).astype(np.int64)
        take = take[pick]
        print(f"  BOUNDED PROBE: {len(take)} of {len(train_idx)} train rows, evenly spaced")

    codes = load_attribute_codes(pcfg, attribute_sizes)[take]
    n = len(codes)

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(ckpt_dir), f"synthetic_w{args.guidance_scale:g}")
    os.makedirs(out_dir, exist_ok=True)
    array_path = os.path.join(out_dir, "resized_expressions.npy")
    progress_path = os.path.join(out_dir, "progress.json")
    shape = (n, config["image_size"], config["image_size"], config["in_channels"])

    # Resume is only honoured when every parameter that changes what a row CONTAINS
    # is identical. Half a run at w=2 stitched to half at w=3 would be a dataset
    # nothing could interpret, and nothing downstream would notice.
    signature = {
        "n": n, "split": args.split, "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps, "checkpoint": os.path.abspath(args.checkpoint),
        "shape": list(shape),
    }
    start = 0
    if os.path.exists(progress_path) and not args.restart:
        prior = json.load(open(progress_path))
        if {k: prior.get(k) for k in signature} == signature and os.path.exists(array_path):
            start = int(prior.get("done", 0))
            print(f"  Resuming at row {start}/{n}")
        else:
            print("  progress.json does not match this run's parameters — starting over")

    print(f"  Output: {out_dir}")
    print(f"  Array : {np.prod(shape) * 4 / 1e9:.1f} GB at {shape}")
    if start >= n:
        print("  Already complete.")
        return

    from diffusion.diffusion_edm_sample import generate_conditioned_batch, load_model

    # load_model reads norm_constants.json out of config['checkpoint_dir']. Point
    # that at the checkpoint's own directory too, so the decode constants and the
    # split record are guaranteed to come from the SAME training run — otherwise a
    # --scratch-dir checkpoint silently falls back to [0,1] outputs while the split
    # is read from somewhere else.
    config = dict(config, checkpoint_dir=ckpt_dir)
    model = load_model(config, args.checkpoint)
    occupancy_mask = None
    mask_path = pcfg.pixel_occupancy_mask_path
    if os.path.exists(mask_path):
        import tensorflow as tf
        occupancy_mask = tf.constant(np.load(mask_path).astype(np.float32))
        print(f"  Occupancy mask: {mask_path}")

    mode = "r+" if start > 0 and os.path.exists(array_path) else "w+"
    images = np.lib.format.open_memmap(array_path, mode=mode, dtype=np.float32, shape=shape)

    for i in range(start, n, args.chunk_size):
        sl = slice(i, min(i + args.chunk_size, n))
        images[sl] = generate_conditioned_batch(
            model, config, codes[sl],
            guidance_scale=args.guidance_scale, num_steps=args.num_steps,
            batch_size=args.batch_size, occupancy_mask=occupancy_mask, verbose=False,
        )
        images.flush()
        json.dump({**signature, "done": sl.stop}, open(progress_path, "w"), indent=2)
        print(f"  {sl.stop}/{n} rows written", flush=True)

    del images
    write_sidecars(out_dir, frame, attribute_sizes, pcfg, take)

    check = np.load(array_path, mmap_mode="r")
    head = np.asarray(check[:min(64, n)], dtype=np.float32)
    print("\n  Sanity:")
    print(f"    finite       : {bool(np.isfinite(head).all())}")
    print(f"    range        : [{head.min():.4f}, {head.max():.4f}]")
    if occupancy_mask is not None:
        off = np.load(mask_path).astype(bool)
        leaked = float(np.abs(head[:, ~off]).max()) if (~off).any() else 0.0
        print(f"    max |value| outside occupied pixels: {leaked:.3e} (want 0)")
    print(f"\nReplica complete: {out_dir}")


if __name__ == "__main__":
    main()
