#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, GlobalAveragePooling2D,
    Activation, Dropout, Dense, LeakyReLU,
)
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing.filter_utils import filter_classes, EXCLUDED_CLASSES
from preprocessing.artifact_paths import DEFAULT_CONFIG

RUN_MODE = os.environ.get("RUN_MODE", "local")
PREPROCESSING_DIR = DEFAULT_CONFIG.artifact_dir
DATA_DIR = f"output/classifier/{RUN_MODE}"
FEATURE_FILE = "resized_expressions.npy"
LABEL_FILE = "y_primary_disease_or_tissue.npy"

MODEL_OUTPUT_FILE = "classifier_small.keras"


def load_data(feature_file, label_file):
    x_train = np.load(feature_file)
    y_train = np.load(label_file)
    num_classes = y_train.shape[1]
    x_train, y_train = filter_classes(x_train, y_train, EXCLUDED_CLASSES)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=1
    )
    return x_train, x_val, y_train, y_val, num_classes


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


def get_model(num_classes, input_shape=DEFAULT_CONFIG.image_shape):
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


def train_model(model, x_train, y_train, x_val, y_val):
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    return model.fit(
        x_train, y_train,
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=[stop_early],
    )


def plot_history(hist):
    plt.plot(hist.epoch, hist.history["loss"])
    plt.plot(hist.epoch, hist.history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (Small)")
    plt.savefig(f"{DATA_DIR}/classifier_small_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to {DATA_DIR}/classifier_small_loss.png")

    plt.plot(hist.epoch, hist.history["accuracy"])
    plt.plot(hist.epoch, hist.history["val_accuracy"])
    plt.legend(["accuracy", "val_accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy (Small)")
    plt.savefig(f"{DATA_DIR}/classifier_small_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Accuracy plot saved to {DATA_DIR}/classifier_small_accuracy.png")


def evaluate_model(model, x_train, y_train, x_val, y_val):
    model.evaluate(x_train, y_train)
    model.evaluate(x_val, y_val)


def save_model(model, weights_only=True):
    file_path = f"{DATA_DIR}/{MODEL_OUTPUT_FILE}"
    if weights_only:
        weights_path = file_path.replace(".keras", "_weights_only.keras")
        model.save(weights_path, save_format="keras", include_optimizer=False)
        print(f"Model weights saved to {weights_path}")
    else:
        model.save(file_path, save_format="keras")
        print(f"Model saved to {file_path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    x_train, x_val, y_train, y_val, num_classes = load_data(
        f"{PREPROCESSING_DIR}/{FEATURE_FILE}",
        f"{PREPROCESSING_DIR}/{LABEL_FILE}",
    )
    model = get_model(num_classes)
    hist = train_model(model, x_train, y_train, x_val, y_val)
    save_model(model)
    plot_history(hist)
    evaluate_model(model, x_train, y_train, x_val, y_val)
