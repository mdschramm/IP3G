from Classifier import load_data, get_model, precision_m, recall_m, f1_m
import tensorflow as tf
import numpy as np 

DATA_DIR = "loaded_data"
MODEL_DIR = "loaded_data_remote"
FEATURE_FILE = "resized_expressions.npy"
LABEL_FILE = "y_primary_disease_or_tissue.npy"

MODEL_WEIGHTS = "classifier_weights_only.keras"

"""
# Confidence Evaluation
Given a trained model and a dataset:
1. Divide the dataset by each class/label
2. For each class:
    2.1. Classify the images in the class using the model
    2.2 The Confidence is the percentage of images that are classified as that class
"""
def evaluate_confidence():

    x_train, x_val, y_train, y_val, num_classes = load_data(f"{DATA_DIR}/{FEATURE_FILE}", f"{DATA_DIR}/{LABEL_FILE}")
    
    # Load model without optimizer state
    weights_path = f"{MODEL_DIR}/{MODEL_WEIGHTS}"
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m}
    )
    print(f"Loaded weights from {weights_path}")
    
    print("Training set evaluation:")
    model.evaluate(x_train, y_train)
    print("\nValidation set evaluation:")
    model.evaluate(x_val, y_val)
    
    # for i in range(num_classes):
    #     class_images = x_train[y_train[:, i] == 1]
    #     class_labels = y_train[y_train[:, i] == 1]
    #     class_predictions = model.predict(class_images)
        
    #     # Get predicted class for each image (argmax across classes)
    #     predicted_classes = np.argmax(class_predictions, axis=1)
    #     most_common = np.bincount(predicted_classes).argmax()
        
    #     # Confidence: percentage of images correctly predicted as class i
    #     confidence = np.mean(predicted_classes == most_common)
        
    #     print(f"Class {i}: most_common={most_common}, confidence={confidence:.4f}, n_samples={len(class_images)}")
    
if __name__ == "__main__":
    evaluate_confidence()
