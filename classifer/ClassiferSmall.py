#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, GlobalAveragePooling2D,
    Activation, Dropout, Dense, LeakyReLU,
)
from tensorflow.keras.optimizers import Adam
import argparse
import json
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from classifer.training_data import ImageBatches, describe_split, make_split
from preprocessing.filter_utils import EXCLUDED_CLASSES
from preprocessing.artifact_paths import GTEX_DATASET, PreprocessingConfig, model_output_dir

RUN_MODE = os.environ.get("RUN_MODE", "local")
RUN_DATASET = os.environ.get("RUN_DATASET") or GTEX_DATASET
FEATURE_FILE = "resized_expressions.npy"

# The label attribute differs by corpus. GTEx carries one fused
# disease-or-tissue array; RNAseqDB carries a separate one-hot per attribute, of
# which tissue is the 15-way analogue.
LABEL_FILE_BY_DATASET = {
    GTEX_DATASET: "y_primary_disease_or_tissue.npy",
    "rnaseqdb": "y_tissue.npy",
}

MODEL_OUTPUT_FILE = "classifier_small.keras"


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    p = precision_m(y_true, y_pred)
    r = recall_m(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def get_model(num_classes, input_shape):
    """
    Smaller classifier for 128×128×16 gene expression images.

    Design decisions vs Classifier.py:
      - 3×3 kernels instead of 15×15 — eliminates the parameter explosion in
        deeper layers (15×15 = 225 weights/pair vs 3×3 = 9).
      - GlobalAveragePooling2D instead of Flatten — avoids the 8×8×256=16,384-
        unit dense bottleneck; pools each feature map to a single scalar instead,
        dramatically reducing the final Dense layer's input size.
      - BatchNormalization after each conv — improves gradient flow and acts as
        a mild regularizer.
      - Filter progression 32→64→128→256 — sufficient capacity for 54 classes
        on ~5,900 training samples without the original 512/768 filter explosion.

    Approximate parameter count:
      conv2d_0 (3×3×16×32):     ~4.6k
      conv2d_1 (3×3×32×64):     ~18.5k
      conv2d_2 (3×3×64×128):    ~73.9k
      conv2d_3 (3×3×128×256):   ~295.2k
      dense_head (256→256→54):  ~79.7k
      BatchNorm params:          ~1.9k
      ─────────────────────────────────
      Total:                     ~474k   (vs 122M in Classifier.py)
    """
    model = Sequential()

    # Block 1 — input_shape → half resolution, 32 filters
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same",
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Block 2 — 64×64×32 → 32×32×64
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Block 3 — 32×32×64 → 16×16×128
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    # Block 4 — 16×16×128 → 8×8×256
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Pool each 8×8 feature map to a scalar → 256-dim vector (no large Dense layer)
    model.add(GlobalAveragePooling2D())

    # Head
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    opt = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy", precision_m, recall_m, f1_m],
    )
    model.summary()
    return model


def train_model(model, train_ds, val_ds, epochs=100, patience=5):
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    return model.fit(train_ds, epochs=epochs, validation_data=val_ds,
                     callbacks=[stop_early])


def plot_history(hist, out_dir):
    for keys, ylabel, fname in (
        (("loss", "val_loss"), "Loss", "classifier_small_loss.png"),
        (("accuracy", "val_accuracy"), "Accuracy", "classifier_small_accuracy.png"),
    ):
        for k in keys:
            plt.plot(hist.epoch, hist.history[k])
        plt.legend(list(keys))
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Training and Validation {ylabel} (Small)")
        plt.savefig(f"{out_dir}/{fname}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {out_dir}/{fname}")


def save_model(model, num_classes, input_shape, out_dir):
    """Weights only, via a freshly built model.

    In Keras 3 `save(..., include_optimizer=False)` is not a Model.save
    parameter — the kwarg is silently swallowed — and plain save_weights() still
    writes the optimizer slots. Adam keeps two per parameter, so either route
    inflates the checkpoint threefold on every save and every rsync off the VM.
    Copying into a model whose optimizer has never stepped leaves no slots to
    write.
    """
    path = os.path.join(out_dir, MODEL_OUTPUT_FILE.replace(".keras", ".weights.h5"))
    export = get_model(num_classes, input_shape)
    export.set_weights(model.get_weights())
    export.save_weights(path)
    print(f"  saved {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def main():
    import pandas as pd

    p = argparse.ArgumentParser(description="Small CNN classifier over expression images")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--split", choices=("stratified", "vinas", "donor"), default="stratified",
                   help="vinas matches the reference paper's procedure; donor removes "
                        "the donor leak that neither their split nor ours otherwise avoids")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap sample count for LOCAL smoke tests only")
    args = p.parse_args()

    config = PreprocessingConfig(args.width, args.height, args.channels, dataset=RUN_DATASET)
    out_dir = model_output_dir("classifier")
    os.makedirs(out_dir, exist_ok=True)

    label_file = LABEL_FILE_BY_DATASET.get(RUN_DATASET)
    if label_file is None:
        raise SystemExit(f"No label file mapped for RUN_DATASET={RUN_DATASET!r}; "
                         f"known: {sorted(LABEL_FILE_BY_DATASET)}")

    print("=" * 70)
    print(f"SMALL CLASSIFIER — {config.dataset} — {config.tag}")
    print("=" * 70)
    print(f"  features : {config.resized_expressions_path}")
    print(f"  labels   : {os.path.join(config.dataset_dir, label_file)}")
    print(f"  outputs  : {out_dir}")

    images = np.load(config.resized_expressions_path, mmap_mode="r")
    y = np.load(os.path.join(config.dataset_dir, label_file)).astype(np.float32)

    # Class exclusion is a GTEx-only concept: those indices name GTEx body sites
    # with too few samples. RNAseqDB keeps every class (empty EXCLUDED_CLASSES).
    keep = np.arange(len(y))
    if RUN_DATASET == GTEX_DATASET and EXCLUDED_CLASSES:
        keep = keep[~np.isin(y.argmax(1), EXCLUDED_CLASSES)]
        print(f"  excluded GTEx classes {EXCLUDED_CLASSES}: {len(keep)} samples remain")

    if os.path.exists(config.labels_path):
        frame = pd.read_csv(config.labels_path).iloc[keep].reset_index(drop=True)
    else:  # GTEx has no label frame; synthesise the minimum make_split needs
        frame = pd.DataFrame({"sample_id": [f"S{i}" for i in keep]})

    if args.max_samples and args.max_samples < len(keep):
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(keep), args.max_samples, replace=False))
        keep, frame = keep[sel], frame.iloc[sel].reset_index(drop=True)
        print(f"  SMOKE TEST: capped to {len(keep)} samples")

    images = images[keep] if len(keep) != len(y) else images
    y = y[keep]
    num_classes = y.shape[1]
    print(f"  images   : {images.shape}   classes: {num_classes}")

    train_idx, val_idx = make_split({"tissue": y}, frame, mode=args.split)
    print(describe_split(frame, train_idx, val_idx, args.split))

    model = get_model(num_classes, config.image_shape)
    train_ds = ImageBatches(images, y, train_idx, args.batch_size, shuffle=True)
    val_ds = ImageBatches(images, y, val_idx, args.batch_size)

    hist = train_model(model, train_ds, val_ds, args.epochs, args.patience)
    save_model(model, num_classes, config.image_shape, out_dir)
    plot_history(hist, out_dir)

    scores = model.evaluate(val_ds, verbose=0, return_dict=True)
    print("\nValidation:")
    for k, v in scores.items():
        print(f"  {k:<14} {v:.4f}")
    with open(os.path.join(out_dir, "classifier_small_metrics.json"), "w") as fh:
        json.dump({"dataset": config.dataset, "tag": config.tag, "split": args.split,
                   "num_classes": int(num_classes), "validation": scores}, fh, indent=2)
    print(f"  saved {out_dir}/classifier_small_metrics.json")


if __name__ == "__main__":
    main()
