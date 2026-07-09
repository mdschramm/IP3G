from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.utils import to_categorical

import numpy as np


from sklearn.metrics import adjusted_rand_score , adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
import keras
import tensorflow as tf
from PIL import Image
import math


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np

from tensorflow.keras.utils import to_categorical
import math
import os
from IPython.display import clear_output
from collections import Counter
import json
import time

BASE_DATA_DIR = "output/preprocessing"

# List of samples and their features and phenotypes
GTEX_PHENOTYPE = "GTEX_phenotype"
# List of gene expressions across samples
RSEM_HUGO_NORM_COUNT = "gtex_RSEM_Hugo_norm_count"

def load_if_not_exists(file_path, calculate, force=False, **kwargs):
  """
  Cache wrapper that calculates and saves data if it doesn't exist, otherwise loads from file.
  
  Args:
    file_path: Path to cache file (.json or .npy)
    calculate: Function to call if cache doesn't exist
    force: If True, recalculate even if cache exists
    **kwargs: Arguments passed to calculate function
    
  Returns:
    Data loaded from cache or calculated by calculate function
    Shape depends on the calculate function used
  """
  use_json = os.path.splitext(file_path)[1] == ".json"
  load_fn = json.load if use_json else np.load
  save_fn = json.dump if use_json else np.save
  if os.path.exists(file_path) and not force:
    mode = "r" if use_json else "rb"
    with open(file_path, mode) as f:
      return load_fn(f)
  else:
    start_time = time.time()
    data = calculate(**kwargs)
    print(f"Ran in in {time.time() - start_time} seconds")
    try:
      mode = "w" if use_json else "wb"
      print(f"Writing to {file_path}")
      with open(file_path, mode) as f:
        if use_json:
          save_fn(data, f)
        else:
          save_fn(f, data)
    except Exception as e:
      print(f"Failed to write to {file_path}: {e}")
      # Remove file if it exists
      if os.path.exists(file_path):
        os.remove(file_path)

      raise
    return data

def load_samples():
  """
  Load sample IDs from the header row of gene expression file.
  
  Reads: First line of RSEM_HUGO_NORM_COUNT file
  Format: gene_id\tsample1\tsample2\t...\tsampleN
  
  Returns:
    np.ndarray: Sample IDs, shape (N_samples,)
    Example: ['GTEX-1117F-0226-SM-5GZZ7', 'GTEX-1117F-0426-SM-5EGHI', ...]
  """
  print(f"Loading samples data from {RSEM_HUGO_NORM_COUNT}")
  with open(RSEM_HUGO_NORM_COUNT, "r") as f:
    samples = f.readline()
    samples = np.asarray(samples.split('\t')[1:])
    return np.char.strip(samples)

def generate_phenotype_mapping(source_file=GTEX_PHENOTYPE, target_column=1):
  """
  Create mapping from sample ID to phenotype (body site).
  
  Reads: GTEX_PHENOTYPE file (tab-separated)
  Format: sample_id\tbody_site\t...
  
  Args:
    source_file: Path to phenotype file
    target_column: Column index for phenotype (1 = body site)
    
  Returns:
    dict: {sample_id: phenotype}
    Example: {'GTEX-1117F-0226-SM-5GZZ7': 'Brain - Cortex', ...}
  """
  print(f"Reading data from {source_file}")
  with open(source_file, "r") as f:
    f.readline()
    _dict = {}
    for line in f:
      fileds = line.split('\t')
      _dict[fileds[0].strip()] = fileds[target_column].strip()
    return _dict

def get_phenotypes(samples, sample_to_phenotype):
  """
  Map sample IDs to their phenotypes using the phenotype mapping.
  
  Args:
    samples: np.ndarray of sample IDs, shape (N_samples,)
    sample_to_phenotype: dict mapping sample_id -> phenotype
    
  Returns:
    np.ndarray: Phenotype labels for each sample, shape (N_samples,)
    Example: ['Brain - Cortex', 'Muscle - Skeletal', 'NA', ...]
    Note: 'NA' indicates sample not found in phenotype mapping
  """
  print("Generating phenotypes list from samples and sample mapping")
  return np.vectorize(lambda x: sample_to_phenotype.get(x, "NA"))(samples)


def calculate_data():
  """
  Load gene expression data and filter out low-expression genes.
  
  Reads: RSEM_HUGO_NORM_COUNT file (tab-separated)
  Format: gene_id\texpr_sample1\texpr_sample2\t...\texpr_sampleN
  
  Processing:
    1. Read each gene (row) with expression values across all samples
    2. Filter: Keep only genes where mean expression >= 0.1
    3. Transpose: Convert from (genes, samples) to (samples, genes)
  
  Returns:
    np.ndarray: Gene expression matrix, shape (N_samples, N_filtered_genes)
    - N_samples: Total number of samples (matches sample IDs from load_samples)
    - N_filtered_genes: Number of genes passing mean >= 0.1 threshold
    
  IMPORTANT: This filters GENES (features), not SAMPLES.
  The sample dimension is preserved, so phenotype labels remain aligned.
  """
  print(f"Filling data from {RSEM_HUGO_NORM_COUNT}")
  with open(RSEM_HUGO_NORM_COUNT, "r") as f:
    f.readline()
    data = []
    i = 0
    for line in f:
      fileds = line.split('\t')
      values = [float(x.strip()) for x in fileds[1:]]
      values = np.asarray(values)
      if values.mean() >= 0.1:
        data.append(values)
        i += 1
        if i % 10000 == 0:
          print(f"Added {i} rows to data")

    data = np.asarray(data)
    return data.T


def get_y_train(phenotypes):
  """
  Convert phenotype labels to one-hot encoded vectors.
  
  Args:
    phenotypes: Array of phenotype labels, shape (N_samples,)
    Example: ['Brain - Cortex', 'Muscle - Skeletal', ...]
    
  Returns:
    np.ndarray: One-hot encoded labels, shape (N_samples, N_classes)
    Each row is a binary vector with 1 at the class index.
  """
  # CRITICAL FIX: Sort labels to ensure deterministic class-to-index mapping
  # Without sorting, set() order is non-deterministic and can change between runs
  # This was causing the 96% -> 2% accuracy drop!
  set_labels = sorted(set(phenotypes))
  num_classes = len(set_labels)
  dict_labels = {}
  for i,l in enumerate(set_labels):
    dict_labels[l] = i
  labels = []
  for l in phenotypes:
    labels.append(dict_labels[l])
  y_train = to_categorical(labels , num_classes=num_classes)
  return y_train


def print_class_label_mapping(y_file_path):
  """
  Print the mapping from class label index (e.g. "class 40") to the
  phenotype value it corresponds to, for a one-hot label file produced
  by get_y_train (y_primary_disease_or_tissue.npy or y_primary_site.npy).

  The class index for a label is determined in get_y_train by
  sorted(set(phenotypes)), so this reloads the same per-sample phenotypes
  array used to build the y file and re-derives that sorted order.

  Args:
    y_file_path: Path to y_primary_disease_or_tissue.npy or y_primary_site.npy
      (as saved by preprocessing.prepare_training_data)
  """
  filename = os.path.basename(y_file_path)
  if "primary_disease_or_tissue" in filename:
    phenotypes_path = f"{BASE_DATA_DIR}/sample_body_site_phenotypes.npy"
  elif "primary_site" in filename:
    phenotypes_path = f"{BASE_DATA_DIR}/sample_primary_site_phenotypes.npy"
  else:
    raise ValueError(f"Unrecognized y label file: {filename}")

  y = np.load(y_file_path)
  phenotypes = np.load(phenotypes_path)
  set_labels = sorted(set(phenotypes))

  if len(set_labels) != y.shape[1]:
    raise ValueError(
      f"Mismatch: {phenotypes_path} has {len(set_labels)} unique phenotypes "
      f"but {y_file_path} has {y.shape[1]} classes. Cached files may be out of sync."
    )

  for i, label in enumerate(set_labels):
    print(f"class {i}: {label}")


if __name__ == "__main__":
  import os
  os.makedirs(BASE_DATA_DIR, exist_ok=True)

  samples = load_if_not_exists(f"{BASE_DATA_DIR}/samples.npy", load_samples)
  sample_count = len(samples)
  print(f"Found {sample_count} total samples")

  phenotype_mapping = load_if_not_exists(f"{BASE_DATA_DIR}/sample_to_body_site_mapping.json", generate_phenotype_mapping)

  sample_body_site_phenotypes = load_if_not_exists(f"{BASE_DATA_DIR}/sample_body_site_phenotypes.npy",
  get_phenotypes,
  samples=samples,
  sample_to_phenotype=phenotype_mapping)

  print("Number of 'NA' phenotypes: ", np.sum(sample_body_site_phenotypes == "NA"))

  data = load_if_not_exists(f"{BASE_DATA_DIR}/data.npy", calculate_data)

  print(data.shape , np.min(data) , np.max(data) ,np.mean(data))