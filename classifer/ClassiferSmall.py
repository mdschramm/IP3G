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


def final_report(model, val_ds, y_true_onehot):
    """AUC and macro-F1 on the validation set, sklearn-style.

    The compiled Keras metrics above cannot produce either column of Viñas et
    al. §5.2.2. `f1_m` sums TP/FP over the whole batch, and under a one-hot
    softmax every sample contributes exactly one predicted positive and one true
    positive — so it is micro-F1, which is arithmetically identical to accuracy
    and systematically higher than the macro-F1 the paper reports whenever the
    rare classes are the hard ones. There is no AUC at all.

    The multiclass-vs-binary branch here is copied deliberately from
    evaluation/vinas_metrics.py:tstr_scores, so a CNN number and an MLP number
    from that harness mean the same thing and can sit in the same table.

    Args:
        y_true_onehot: labels in the exact order val_ds yields them.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    proba = model.predict(val_ds, verbose=0)
    y_true = y_true_onehot.argmax(1)
    pred = proba.argmax(1)
    n_classes = y_true_onehot.shape[1]

    if n_classes == 2:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        # Mean one-vs-rest AUC over the classes actually present. When every class
        # is present this is exactly roc_auc_score(multi_class="ovr",
        # average="macro") — sklearn averages these same per-class binary AUCs —
        # but that call raises outright if a class is missing from y_true, which
        # turns a capped smoke run into a crash instead of a number.
        aucs = [roc_auc_score((y_true == c).astype(int), proba[:, c])
                for c in np.unique(y_true)]
        auc = float(np.mean(aucs))
    return {
        "accuracy": float((pred == y_true).mean()),
        "auc": float(auc),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted")),
    }


def train_model(model, train_ds, val_ds, epochs=100, patience=5):
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    return model.fit(train_ds, epochs=epochs, validation_data=val_ds,
                     callbacks=[stop_early])


def plot_history(hist, out_dir, suffix=""):
    for keys, ylabel, fname in (
        (("loss", "val_loss"), "Loss", f"classifier_small_loss{suffix}.png"),
        (("accuracy", "val_accuracy"), "Accuracy", f"classifier_small_accuracy{suffix}.png"),
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


def save_model(model, num_classes, input_shape, out_dir, suffix=""):
    """Weights only, via a freshly built model.

    In Keras 3 `save(..., include_optimizer=False)` is not a Model.save
    parameter — the kwarg is silently swallowed — and plain save_weights() still
    writes the optimizer slots. Adam keeps two per parameter, so either route
    inflates the checkpoint threefold on every save and every rsync off the VM.
    Copying into a model whose optimizer has never stepped leaves no slots to
    write.
    """
    path = os.path.join(out_dir, MODEL_OUTPUT_FILE.replace(".keras", f"{suffix}.weights.h5"))
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
    p.add_argument("--attribute", choices=("tissue", "condition"), default=None,
                   help="rnaseqdb only: which attribute to classify. Viñas et al. "
                        "§5.2.2 reports both a 15-way tissue row and a binary "
                        "cancer/normal row. Default: the corpus's usual label array.")
    p.add_argument("--synthetic-dir", default=None,
                   help="train on the synthetic replica in this directory instead of "
                        "the real training split. Validation ALWAYS stays on real data.")
    p.add_argument("--runs", type=int, default=1,
                   help="independent re-inits, reported as mean +/- std (paper uses 5)")
    p.add_argument("--out-suffix", default=None,
                   help="tag appended to every output filename so the real-trained and "
                        "synthetic-trained runs do not overwrite each other")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap sample count for LOCAL smoke tests only")
    args = p.parse_args()
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""

    config = PreprocessingConfig(args.width, args.height, args.channels, dataset=RUN_DATASET)
    out_dir = model_output_dir("classifier")
    os.makedirs(out_dir, exist_ok=True)

    if args.attribute:
        if RUN_DATASET == GTEX_DATASET:
            raise SystemExit("--attribute is an rnaseqdb concept: GTEx carries one fused "
                             "disease-or-tissue array, not a one-hot per attribute.")
        label_path = config.y_attribute_path(args.attribute)
    else:
        label_file = LABEL_FILE_BY_DATASET.get(RUN_DATASET)
        if label_file is None:
            raise SystemExit(f"No label file mapped for RUN_DATASET={RUN_DATASET!r}; "
                             f"known: {sorted(LABEL_FILE_BY_DATASET)}")
        label_path = os.path.join(config.dataset_dir, label_file)
    label_basename = os.path.basename(label_path)

    print("=" * 70)
    print(f"SMALL CLASSIFIER — {config.dataset} — {config.tag}"
          f"{' — ' + args.attribute if args.attribute else ''}")
    print("=" * 70)
    print(f"  features : {config.resized_expressions_path}")
    print(f"  labels   : {label_path}")
    print(f"  outputs  : {out_dir}  (suffix {suffix!r})")

    images = np.load(config.resized_expressions_path, mmap_mode="r")
    y = np.load(label_path).astype(np.float32)

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

    # Sorted so the concatenation of val batches is in a known order: ImageBatches
    # sorts each batch internally for memmap locality, which would otherwise
    # permute predictions relative to y[val_idx] and quietly scramble the report.
    val_idx = np.sort(val_idx)
    y_val = y[val_idx]

    if args.synthetic_dir:
        # The replica IS the training split — every row of it is used, and the
        # validation set stays on real arrays. That asymmetry is the entire point
        # of TSTR, so it is not configurable.
        syn_x = np.load(os.path.join(args.synthetic_dir, FEATURE_FILE), mmap_mode="r")
        syn_y = np.load(os.path.join(args.synthetic_dir, label_basename)).astype(np.float32)
        if len(syn_x) != len(syn_y):
            raise SystemExit(f"{args.synthetic_dir}: {len(syn_x)} images but {len(syn_y)} labels")
        if syn_y.shape[1] != num_classes:
            raise SystemExit(f"{args.synthetic_dir}: {syn_y.shape[1]} classes, real data has "
                             f"{num_classes}")
        if len(syn_x) != len(train_idx):
            msg = (f"synthetic replica has {len(syn_x)} rows but the '{args.split}' train "
                   f"split has {len(train_idx)}")
            if not args.max_samples:
                raise SystemExit(msg + " — it was built against a different partition.")
            print(f"  NOTE: {msg} (expected under --max-samples)")
        train_x, train_y, train_sel = syn_x, syn_y, np.arange(len(syn_y))
        trained_on = f"synthetic:{args.synthetic_dir}"
        print(f"  TSTR: training on {len(syn_y)} synthetic rows, validating on "
              f"{len(val_idx)} REAL rows")
    else:
        train_x, train_y, train_sel = images, y, train_idx
        trained_on = "real"

    train_ds = ImageBatches(train_x, train_y, train_sel, args.batch_size, shuffle=True)
    val_ds = ImageBatches(images, y, val_idx, args.batch_size)

    runs = []
    for r in range(args.runs):
        print(f"\n--- run {r + 1}/{args.runs} ---")
        tf.keras.utils.set_random_seed(r)
        model = get_model(num_classes, config.image_shape)
        hist = train_model(model, train_ds, val_ds, args.epochs, args.patience)
        keras_scores = model.evaluate(val_ds, verbose=0, return_dict=True)
        report = final_report(model, val_ds, y_val)
        runs.append({**report, "keras": keras_scores})
        print("  " + "  ".join(f"{k}={report[k]:.4f}"
                               for k in ("accuracy", "auc", "f1_macro", "f1_weighted")))
        if r == 0:  # weights and curves from the first run only; the rest are for the spread
            save_model(model, num_classes, config.image_shape, out_dir, suffix)
            plot_history(hist, out_dir, suffix)

    keys = ("accuracy", "auc", "f1_macro", "f1_weighted")
    mean = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    std = {k: float(np.std([r[k] for r in runs])) for k in keys}
    print(f"\nValidation over {args.runs} run(s) — real held-out data:")
    for k in keys:
        print(f"  {k:<14} {mean[k]:.4f} ± {std[k]:.4f}")

    metrics_path = os.path.join(out_dir, f"classifier_small_metrics{suffix}.json")
    with open(metrics_path, "w") as fh:
        json.dump({"dataset": config.dataset, "tag": config.tag, "split": args.split,
                   "attribute": args.attribute, "trained_on": trained_on,
                   "num_classes": int(num_classes), "n_train": int(len(train_sel)),
                   "n_val": int(len(val_idx)), "n_runs": args.runs,
                   "mean": mean, "std": std, "runs": runs}, fh, indent=2)
    print(f"  saved {metrics_path}")


if __name__ == "__main__":
    main()
