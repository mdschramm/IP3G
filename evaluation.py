from Classifier import load_model, load_data

DATA_DIR = "loaded_data"
FEATURE_FILE = "resized_expressions.npy"
LABEL_FILE = "y_primary_disease_or_tissue.npy"

MODEL_OUTPUT = "classifier.keras"

"""
# Confidence Evaluation
Given a trained model and a dataset:
1. Divide the dataset by each class/label
2. For each class:
    2.1. Classify the images in the class using the model
    2.2 The Confidence is the percentage of images that are classified as that class
"""
def evaluate_confidence():
    model = load_model(MODEL_OUTPUT)
    x_train, x_val, y_train, y_val, num_classes = load_data(f"{DATA_DIR}/{FEATURE_FILE}", f"{DATA_DIR}/{LABEL_FILE}")
    print(y_train)
    # for i in range(num_classes):
    #     class_images = x_train[y_train[:, i] == 1]
    #     class_labels = y_train[y_train[:, i] == 1]
    #     class_predictions = model.predict(class_images)
    #     confidence = np.mean(class_predictions[:, i])
    #     print(f"Class {i} confidence: {confidence}")
    
