#!/usr/bin/env python
# coding: utf-8

import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing.filter_utils import filter_classes, EXCLUDED_CLASSES
from preprocessing.artifact_paths import DEFAULT_CONFIG, model_output_dir

RUN_MODE = os.environ.get("RUN_MODE", "local")
PREPROCESSING_DIR = DEFAULT_CONFIG.artifact_dir
DATA_DIR = model_output_dir("classifier")
FEATURE_FILE = "resized_expressions.npy"
LABEL_FILE = "y_primary_disease_or_tissue.npy"

MODEL_OUTPUT_FILE = "classifier.keras"

def load_data(feature_file, label_file):
    x_train =np.load(feature_file)
    y_train=np.load(label_file)
    num_classes = y_train.shape[1]
    x_train, y_train = filter_classes(x_train, y_train, EXCLUDED_CLASSES)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
    return x_train, x_val, y_train, y_val, num_classes


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model(num_classes, input_shape=DEFAULT_CONFIG.image_shape):
    model = Sequential()

    model.add(Conv2D(32 , kernel_size=15 , strides=2 , padding="same" , input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

            
    model.add(Conv2D(256 , kernel_size=15 , strides=2 , padding="same" ))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    model.add(Conv2D(512 , kernel_size=15 , strides=2 , padding="same" ))
    model.add(LeakyReLU(alpha=0.2))

            
    model.add(Conv2D(768 , kernel_size=15 , strides=2 , padding="same" ))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dropout(0.2))

    # softmax classifier
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy",precision_m,recall_m,f1_m] )
    model.summary()
    return model

def train_model(model, x_train, y_train, x_val, y_val):
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5 ,restore_best_weights=True)
    hist = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[stop_early])
    # hist = model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val), callbacks=[stop_early])
    return hist


def plot_history(hist):
    # Save loss plot
    plt.plot(hist.epoch , hist.history['loss'])
    plt.plot(hist.epoch , hist.history['val_loss'])
    plt.legend(['loss','val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(f'{DATA_DIR}/classifier_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {DATA_DIR}/classifier_loss.png")

    # Save accuracy plot
    plt.plot(hist.epoch , hist.history['accuracy'])
    plt.plot(hist.epoch , hist.history['val_accuracy'])
    plt.legend(['accuracy','val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(f'{DATA_DIR}/classifier_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy plot saved to {DATA_DIR}/classifier_accuracy.png")


def evaluate_model(model, x_train, y_train, x_val, y_val):
    model.evaluate(x_train, y_train)
    model.evaluate(x_val, y_val)

def save_model(model, file_path=f"{DATA_DIR}/{MODEL_OUTPUT_FILE}", weights_only=True):
    # Save weights only (no optimizer state) - ~467MB instead of ~1.4GB
    if weights_only:
        weights_path = file_path.replace('.keras', '_weights_only.keras')
        model.save(weights_path, save_format='keras', include_optimizer=False)
        print(f"Model weights saved to {weights_path}")
    else:
        model.save(file_path, save_format='keras')
        print(f"Model saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Conditional DDPM with Classifier-Free Guidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""""
    )
    parser.add_argument(
        '--load-model',
        type=str,
        choices=["yes", "no"],
        default='no',
        help='Load model from checkpoint'
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    x_train, x_val, y_train, y_val, num_classes = load_data(f"{PREPROCESSING_DIR}/{FEATURE_FILE}", f"{PREPROCESSING_DIR}/{LABEL_FILE}")
    model = get_model(num_classes)
    model.summary()

    load_model = True if parser.parse_args().load_model.lower() == "yes" else False
    if load_model:
        model = tf.keras.models.load_model(f"{DATA_DIR}/{MODEL_OUTPUT_FILE}")
    else:
        hist = train_model(model, x_train, y_train, x_val, y_val)
        save_model(model)
        plot_history(hist)
    evaluate_model(model, x_train, y_train, x_val, y_val)
