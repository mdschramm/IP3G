from classifer.Classifier import load_data, get_model, precision_m, recall_m, f1_m
import os
import tensorflow as tf
import numpy as np
import csv
import argparse
from datetime import datetime

RUN_MODE = os.environ.get("RUN_MODE", "local")
PREPROCESSING_DIR = "output/preprocessing"
CLASSIFIER_DIR = f"output/classifier/{RUN_MODE}"
GAN_DIR = f"output/gan/{RUN_MODE}"
DIFFUSION_DIR = f"output/diffusion/{RUN_MODE}"
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
def map_latent_to_phenotype(latent_feature_file, latent_label_file, output_file=None):
    """
    Map GAN's unsupervised latent classes to real phenotypes.
    
    For each latent class discovered by the GAN:
    1. Get all synthetic images from that latent class
    2. Run them through the trained classifier
    3. Find the most common predicted phenotype
    4. This is the mapping: latent_class_X → phenotype_Y
    
    Args:
        latent_feature_file: Path to synthetic images (from GAN)
        latent_label_file: Path to latent class labels (from GAN)
        output_file: Optional CSV file to save mapping
    """
    # Load synthetic data with GAN's latent classes
    X_synthetic = np.load(latent_feature_file)
    y_latent = np.load(latent_label_file)
    
    # Load trained classifier
    weights_path = f"{CLASSIFIER_DIR}/classifier_weights_only.keras"
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m}
    )
    print(f"Loaded classifier from {weights_path}")
    
    num_latent_classes = y_latent.shape[1]
    print(f"\nMapping {num_latent_classes} GAN latent classes to real phenotypes...\n")
    
    results = []
    
    for latent_class in range(num_latent_classes):
        # Get all images from this latent class
        class_mask = y_latent[:, latent_class] == 1
        class_images = X_synthetic[class_mask]
        
        if len(class_images) == 0:
            print(f"Latent class {latent_class}: No samples found")
            continue
        
        # Predict phenotypes using classifier
        predictions = model.predict(class_images, verbose=0)
        predicted_phenotypes = np.argmax(predictions, axis=1)
        
        # Find most common predicted phenotype
        unique, counts = np.unique(predicted_phenotypes, return_counts=True)
        most_common_idx = unique[np.argmax(counts)]
        most_common_count = np.max(counts)
        confidence = most_common_count / len(class_images)
        
        # Calculate distribution across top predictions
        top_3_indices = unique[np.argsort(counts)[-3:][::-1]]
        top_3_counts = counts[np.argsort(counts)[-3:][::-1]]
        
        print(f"Latent class {latent_class:2d}: n={len(class_images):4d} → "
              f"Phenotype {most_common_idx:2d} ({confidence*100:5.1f}%) | "
              f"Top 3: {list(zip(top_3_indices, top_3_counts))}")
        
        results.append({
            'latent_class': latent_class,
            'n_samples': len(class_images),
            'mapped_phenotype': int(most_common_idx),
            'confidence': round(confidence, 4),
            'top_phenotype_count': int(most_common_count)
        })
    
    # Save mapping to CSV if requested
    if output_file:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['latent_class', 'n_samples', 'mapped_phenotype', 'confidence', 'top_phenotype_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nMapping saved to {output_file}")
    
    return results


def evaluate_diffusion_map(feature_file=None, label_file=None, output_file=None, guidance_scale=3.0):
    """
    Evaluate diffusion class fidelity: for each class C, what fraction of images
    generated conditioned on C are classified as C by the tissue classifier?

    Unlike the GAN MAP (which discovers unsupervised latent→phenotype mappings),
    diffusion generation is class-conditioned, so the true label is known. This
    measures how faithfully the model renders each tissue class.

    Args:
        feature_file: Path to diffusion-generated images npy (shape [N, H, W]).
            Default: {DIFFUSION_DIR}/samples/diffusion_synthetic_expressions_w{guidance_scale}.npy
        label_file: Path to one-hot class labels npy (shape [N, num_classes]).
            Default: {DIFFUSION_DIR}/samples/diffusion_synthetic_labels_w{guidance_scale}.npy
        output_file: Optional CSV path to save per-class results.
        guidance_scale: Guidance scale used during generation (determines default filenames).
    """
    sample_dir = os.path.join(DIFFUSION_DIR, "samples")
    feat_path = feature_file or os.path.join(sample_dir, f"diffusion_synthetic_expressions_w{guidance_scale:.1f}.npy")
    label_path = label_file or os.path.join(sample_dir, f"diffusion_synthetic_labels_w{guidance_scale:.1f}.npy")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Diffusion samples not found at {feat_path}.\n"
            f"Run diffusion_sample.py --generate-dataset first:\n"
            f"  python -m diffusion.diffusion_sample --mode {RUN_MODE} "
            f"--checkpoint <path> --generate-dataset --guidance-scale {guidance_scale}"
        )

    X_synthetic = np.load(feat_path)
    y_synthetic = np.load(label_path)

    if X_synthetic.ndim == 3:
        X_synthetic = X_synthetic[..., np.newaxis]  # [N, H, W] → [N, H, W, 1]

    weights_path = f"{CLASSIFIER_DIR}/classifier_weights_only.keras"
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m}
    )
    print(f"Loaded classifier from {weights_path}")

    num_classes = y_synthetic.shape[1]
    print(f"\nEvaluating diffusion class fidelity (guidance scale w={guidance_scale}):")
    print(f"  {len(X_synthetic)} total samples across {num_classes} classes\n")

    results = []

    for class_id in range(num_classes):
        class_mask = y_synthetic[:, class_id] == 1
        class_images = X_synthetic[class_mask]

        if len(class_images) == 0:
            print(f"Class {class_id:2d}: No samples found")
            continue

        predictions = model.predict(class_images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        fidelity = np.mean(predicted_classes == class_id)

        unique, counts = np.unique(predicted_classes, return_counts=True)
        order = np.argsort(counts)[::-1]
        top3 = list(zip(unique[order[:3]].tolist(), counts[order[:3]].tolist()))

        print(f"Class {class_id:2d}: n={len(class_images):4d} → "
              f"Fidelity {fidelity*100:5.1f}% | Top-3 predicted: {top3}")

        results.append({
            'class': class_id,
            'n_samples': len(class_images),
            'fidelity': round(float(fidelity), 4),
        })

    mean_fidelity = np.mean([r['fidelity'] for r in results])
    print(f"\nMean class fidelity: {mean_fidelity*100:.1f}%")

    if output_file is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(DIFFUSION_DIR, f"diffusion_map_w{guidance_scale:.1f}_{timestamp}.csv")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['class', 'n_samples', 'fidelity'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_file}")

    return results


def evaluate_accuracy(feature_file=None, label_file=None):
    # Use provided files or defaults
    feat_path = feature_file if feature_file else f"{PREPROCESSING_DIR}/{FEATURE_FILE}"
    label_path = label_file if label_file else f"{PREPROCESSING_DIR}/{LABEL_FILE}"

    x_train, x_val, y_train, y_val, num_classes = load_data(feat_path, label_path)

    weights_path = f"{CLASSIFIER_DIR}/{MODEL_WEIGHTS}"
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m}
    )
    print(f"Loaded weights from {weights_path}")
    
    print("Training set evaluation:")
    model.evaluate(x_train, y_train)
    print("\nValidation set evaluation:")
    model.evaluate(x_val, y_val)

def evaluate_confidence(output_file=None, feature_file=None, label_file=None):
    # Use provided files or defaults
    feat_path = feature_file if feature_file else f"{PREPROCESSING_DIR}/{FEATURE_FILE}"
    label_path = label_file if label_file else f"{PREPROCESSING_DIR}/{LABEL_FILE}"

    x_train, x_val, y_train, y_val, num_classes = load_data(feat_path, label_path)

    weights_path = f"{CLASSIFIER_DIR}/classifier_weights_only.keras"
    model = tf.keras.models.load_model(
        weights_path,
        custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m}
    )
    print(f"Loaded weights from {weights_path}")

    if output_file:
        csv_filename = output_file
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"{CLASSIFIER_DIR}/class-confidence-{timestamp}.csv"
    
    results = []
    
    for i in range(num_classes):
        class_images = x_train[y_train[:, i] == 1]
        class_predictions = model.predict(class_images)
        
        # Get predicted class for each image (argmax across classes)
        predicted_classes = np.argmax(class_predictions, axis=1)
        most_common = np.bincount(predicted_classes).argmax()
        
        # Confidence: percentage of images correctly predicted as class i
        confidence = np.mean(predicted_classes == most_common)
        
        # Accuracy: percentage of images correctly predicted as their true class
        true_classes = np.argmax(y_train[y_train[:, i] == 1], axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        
        n_samples = len(class_images)
        print(f"Class {i}: most_common={most_common}, confidence={confidence:.4f}, accuracy={accuracy:.4f}, n_samples={n_samples}")
        
        results.append({
            'class': i,
            'most_common_predicted': most_common,
            'confidence': round(confidence, 4),
            'accuracy': round(accuracy, 4),
            'n_samples': n_samples
        })
    
    # Write results to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['class', 'most_common_predicted', 'confidence', 'accuracy', 'n_samples']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {csv_filename}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate classifier model and map GAN latent classes')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['accuracy', 'confidence', 'map'],
        default='accuracy',
        help='Evaluation mode: accuracy, confidence, or map (default: accuracy)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for CSV results'
    )
    parser.add_argument(
        '--feature-file',
        type=str,
        default=None,
        help=f'Path to feature file (default: {PREPROCESSING_DIR}/{FEATURE_FILE})'
    )
    parser.add_argument(
        '--label-file',
        type=str,
        default=None,
        help=f'Path to label file (default: {PREPROCESSING_DIR}/{LABEL_FILE})'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['gan', 'diffusion'],
        default='gan',
        help='Model source for --mode map: "gan" maps unsupervised latent classes to phenotypes; '
             '"diffusion" measures class fidelity of conditioned generation (default: gan)'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=3.0,
        help='Guidance scale used when generating diffusion samples (affects default filenames, default: 3.0)'
    )

    args = parser.parse_args()

    if args.mode == 'map':
        if args.source == 'diffusion':
            evaluate_diffusion_map(
                feature_file=args.feature_file,
                label_file=args.label_file,
                output_file=args.output,
                guidance_scale=args.guidance_scale,
            )
        else:
            # GAN: map unsupervised latent classes to real phenotypes
            feature_file = args.feature_file or f'{GAN_DIR}/synthetic_resized_expressions.npy'
            label_file = args.label_file or f'{GAN_DIR}/synthetic_latent_classes.npy'
            output_file = args.output or f'{GAN_DIR}/latent_to_phenotype_mapping.csv'

            map_latent_to_phenotype(
                latent_feature_file=feature_file,
                latent_label_file=label_file,
                output_file=output_file
            )
    elif args.mode == 'confidence':
        evaluate_confidence(
            output_file=args.output,
            feature_file=args.feature_file,
            label_file=args.label_file
        )
    elif args.mode == 'accuracy':
        evaluate_accuracy(
            feature_file=args.feature_file,
            label_file=args.label_file
        )
