#!/usr/bin/env python
"""
Multi-attribute tissue/condition classifier for the RNAseqDB corpus.

WHY THIS IS A SEPARATE MODULE FROM Classifier.py
    The GTEx classifier is Sequential with one softmax, and evaluation.py consumes
    it as `np.argmax(model.predict(x), axis=1)` — a single array. A multi-output
    model returns a dict and would break that call site silently. Rather than
    convert a validated model and its downstream consumers in the same change,
    this adds the multi-head variant alongside. The conv trunk is deliberately
    identical to Classifier.get_model so the two remain comparable.

WHAT IT MEASURES
    One softmax head per label attribute over a shared trunk, so evaluation
    decomposes: a model that nails tissue but coin-flips cancer status is a
    specific, diagnosable failure that a single accuracy number hides.

    tissue     15-way   anatomical site
    condition   2-way   normal / tumor
    subtype    20-way   19 TCGA codes + "none" for every normal
    source      2-way   gtex / tcga  -- a DIAGNOSTIC head, not a goal. GTEx is
                        100% normal, so source and condition are near-collinear
                        and a condition head may partly be learning study of
                        origin. If the source head separates GTEx from TCGA
                        *among normals only*, ComBat left residual batch effect.
                        Reported by --report-slices.

USAGE
    RUN_DATASET=rnaseqdb python -m classifer.MultiHeadClassifier
    RUN_DATASET=rnaseqdb python -m classifer.MultiHeadClassifier --epochs 5 --heads tissue,condition
    RUN_DATASET=rnaseqdb python -m classifer.MultiHeadClassifier --report-slices
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from preprocessing.artifact_paths import PreprocessingConfig, model_output_dir
from preprocessing.label_frame import load_vocab

DEFAULT_HEADS = ("tissue", "condition", "subtype", "source")
MODEL_OUTPUT_FILE = "multihead_classifier.keras"


class ImageBatches(tf.keras.utils.PyDataset):
    """Batch generator over a memory-mapped image array.

    resized_expressions.npy is 9.6 GB for this corpus. Loading it and then
    calling train_test_split copies it, peaking near 19 GB — enough to thrash a
    16 GB machine. Indexing a memmap per batch keeps resident memory at roughly
    batch size instead, and costs nothing on a large VM.
    """

    def __init__(self, images, labels, indices, batch_size, shuffle=False, seed=0, **kw):
        super().__init__(**kw)
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        if shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, i):
        # sorted so the memmap is read close to sequentially
        idx = np.sort(self.indices[i * self.batch_size:(i + 1) * self.batch_size])
        x = np.asarray(self.images[idx], dtype=np.float32)
        y = {h: v[idx] for h, v in self.labels.items()}
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def build_model(head_sizes, input_shape, learning_rate=1e-4, loss_weights=None):
    """Shared conv trunk (identical to Classifier.get_model) with one head per attribute."""
    inp = layers.Input(shape=input_shape, name="image")
    x = inp
    for filters in (32, 256, 512, 768):
        x = layers.Conv2D(filters, 15, strides=2, padding="same")(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        if filters in (256, 768):
            x = layers.Dropout(0.5)(x)
    x = layers.Dropout(0.2)(layers.Flatten()(x))

    outputs = {name: layers.Dense(n, activation="softmax", name=name)(x)
               for name, n in head_sizes.items()}

    model = tf.keras.Model(inp, outputs, name="multihead_classifier")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={n: "categorical_crossentropy" for n in head_sizes},
        # Uniform to start. Tissue converges fastest and will otherwise dominate
        # the gradient before the condition head has learned anything.
        loss_weights=loss_weights or {n: 1.0 for n in head_sizes},
        metrics={n: ["accuracy"] for n in head_sizes},
    )
    return model


def stratified_split(labels, heads, test_size=0.25, seed=1):
    """Split stratified on tissue x condition, not on a single attribute.

    Some cells are genuinely thin — cervix has 13 normals — so an unstratified
    split can leave a cell entirely absent from one side and make its per-class
    accuracy undefined rather than merely noisy.
    """
    n = len(next(iter(labels.values())))
    strat = None
    if "tissue" in labels and "condition" in labels:
        joint = (labels["tissue"].argmax(1).astype(np.int64) * 100
                 + labels["condition"].argmax(1))
        counts = np.bincount(joint)
        if counts[counts > 0].min() >= 2:
            strat = joint
    return train_test_split(np.arange(n), test_size=test_size,
                            random_state=seed, stratify=strat)


def report_slices(model, images, labels, val_idx, meta, heads):
    """Per-head accuracy overall, plus the two slices that control for confounds."""
    ds = ImageBatches(images, labels, val_idx, batch_size=64)
    preds = model.predict(ds, verbose=0)
    if not isinstance(preds, dict):
        preds = {heads[0]: preds}

    truth = {h: labels[h][np.sort(val_idx)].argmax(1) for h in heads}
    pred = {h: preds[h].argmax(1) for h in heads}
    order = np.sort(val_idx)

    print("\n" + "=" * 70)
    print("PER-HEAD ACCURACY (validation)")
    print("=" * 70)
    for h in heads:
        print(f"  {h:<12} {np.mean(pred[h] == truth[h]):.4f}   ({len(order)} samples)")

    if "condition" in heads:
        src = meta["source"][order]
        m = src == "tcga"
        if m.any():
            print(f"\n  condition, TCGA-only slice : "
                  f"{np.mean(pred['condition'][m] == truth['condition'][m]):.4f}  ({m.sum()} samples)")
            print("    GTEx is 100% normal, so overall condition accuracy is inflated by "
                  "study\n    of origin. This slice is the honest number.")
    if "source" in heads and "condition" in heads:
        cond = meta["condition"][order]
        m = cond == "normal"
        if m.any():
            acc = np.mean(pred["source"][m] == truth["source"][m])
            print(f"\n  source, normals-only slice : {acc:.4f}  ({m.sum()} samples)")
            print("    High here means ComBat left residual batch effect that a condition")
            print("    head could be exploiting. Near chance means the correction held.")
    return {h: float(np.mean(pred[h] == truth[h])) for h in heads}


def main():
    p = argparse.ArgumentParser(description="Multi-head RNAseqDB classifier")
    p.add_argument("--heads", default=",".join(DEFAULT_HEADS))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--report-slices", action="store_true",
                   help="print the confound-control slices after training")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap the sample count. For LOCAL smoke tests only: the full "
                        "9.6 GB image array cannot stay in page cache alongside TF on a "
                        "16 GB machine, so shuffled batch reads thrash the disk. Not "
                        "needed on the VM, where the array fits in RAM.")
    args = p.parse_args()

    heads = tuple(h.strip() for h in args.heads.split(",") if h.strip())
    config = PreprocessingConfig(args.width, args.height, args.channels,
                                 dataset=os.environ.get("RUN_DATASET") or "gtex")
    out_dir = model_output_dir("classifier")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"MULTI-HEAD CLASSIFIER — {config.dataset} — {config.tag}")
    print("=" * 70)
    print(f"  features : {config.resized_expressions_path}")
    print(f"  outputs  : {out_dir}")

    images = np.load(config.resized_expressions_path, mmap_mode="r")
    labels = {h: np.load(config.y_attribute_path(h)).astype(np.float32) for h in heads}
    vocab = load_vocab(config.attribute_vocab_path)

    import pandas as pd
    frame = pd.read_csv(config.labels_path)

    if args.max_samples and args.max_samples < len(images):
        # Stratified subsample so every tissue x condition cell survives the cap.
        keep = (frame.groupby(["tissue", "condition"], group_keys=False)
                     .apply(lambda g: g.sample(
                         max(2, int(round(args.max_samples * len(g) / len(frame)))),
                         random_state=0))
                     .index.to_numpy())
        keep = np.sort(keep)
        images = images[keep]
        labels = {h: v[keep] for h, v in labels.items()}
        frame = frame.iloc[keep].reset_index(drop=True)
        print(f"  SMOKE TEST: capped to {len(keep)} samples (stratified)")

    head_sizes = {h: labels[h].shape[1] for h in heads}
    meta = {c: frame[c].to_numpy() for c in frame.columns}
    print(f"  images   : {images.shape}")
    print(f"  heads    : {head_sizes}")

    train_idx, val_idx = stratified_split(labels, heads)
    print(f"  split    : {len(train_idx)} train / {len(val_idx)} val (stratified on tissue x condition)")

    model = build_model(head_sizes, config.image_shape, args.learning_rate)
    model.summary()

    train_ds = ImageBatches(images, labels, train_idx, args.batch_size, shuffle=True)
    val_ds = ImageBatches(images, labels, val_idx, args.batch_size)

    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                            restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[stop])

    # Saving needs care in Keras 3. `save(..., include_optimizer=False)` is not a
    # Model.save parameter there — the kwarg is silently swallowed — and plain
    # save_weights() still writes the optimizer slots. Adam keeps 2 slots per
    # parameter, so either route triples this 121M-param model to ~1.45 GB, on
    # every checkpoint and through every rsync off the VM.
    #
    # Copying into a freshly built model whose optimizer has never stepped means
    # no slot variables exist to write, so only the trained weights land on disk.
    weights_path = os.path.join(out_dir, MODEL_OUTPUT_FILE.replace(".keras", ".weights.h5"))
    export_model = build_model(head_sizes, config.image_shape, args.learning_rate)
    export_model.set_weights(model.get_weights())
    export_model.save_weights(weights_path)
    with open(os.path.join(out_dir, "multihead_head_sizes.json"), "w") as fh:
        json.dump({"head_sizes": head_sizes, "image_shape": list(config.image_shape)}, fh)
    print(f"\nSaved {weights_path}  ({os.path.getsize(weights_path)/1e6:.0f} MB)")
    print("  rebuild with build_model(head_sizes, image_shape) then load_weights()")

    if args.report_slices:
        acc = report_slices(model, images, labels, val_idx, meta, heads)
        with open(os.path.join(out_dir, "multihead_accuracy.json"), "w") as fh:
            json.dump({"per_head_accuracy": acc, "vocab_sizes": head_sizes}, fh, indent=2)
        print(f"Saved {out_dir}/multihead_accuracy.json")


if __name__ == "__main__":
    main()
