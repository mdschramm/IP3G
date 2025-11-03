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

BASE_DATA_DIR = "loaded_data"

# List of samples and their features and phenotypes
GTEX_PHENOTYPE = "GTEX_phenotype"
# List of gene expressions across samples
RSEM_HUGO_NORM_COUNT = "gtex_RSEM_Hugo_norm_count"

# Calculates data using calculate function and saves it to file_path if it doesn't exist
# otherwise loads it from file_path
# kwargs are passed to the calculate function
# example usage:
# load_if_not_exists("loaded_data/samples.json", load_samples)
def load_if_not_exists(file_path, calculate, **kwargs):
  use_json = os.path.splitext(file_path)[1] == ".json"
  load_fn = json.load if use_json else np.load
  save_fn = json.dump if use_json else np.save
  if os.path.exists(file_path):
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

# Loads sample data
def load_samples():
  print(f"Loading samples data from {RSEM_HUGO_NORM_COUNT}")
  with open(RSEM_HUGO_NORM_COUNT, "r") as f:
    samples = f.readline()
    samples = np.asarray(samples.split('\t')[1:])
    return np.char.strip(samples)

# Generates the dict object, used once
def generate_phenotype_mapping(source_file=GTEX_PHENOTYPE, target_column=1):
  print(f"Reading data from {source_file}")
  with open(source_file, "r") as f:
    f.readline()
    _dict = {}
    for line in f:
      fileds = line.split('\t')
      _dict[fileds[0].strip()] = fileds[target_column].strip()
    return _dict

# Get phenotypes list from samples and sample_to_phenotype mapping
def get_phenotypes(samples, sample_to_phenotype):
  print("Generating phenotypes list from samples and sample mapping")
  return np.vectorize(lambda x: sample_to_phenotype.get(x, "NA"))(samples)


def calculate_data():
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


if __name__ == "__main__":
  # Get list of samples
  samples = load_if_not_exists("loaded_data/samples.npy", load_samples)
  sample_count = len(samples)
  print(f"Found {sample_count} total samples")
  
  # Map phenotypes to samples
  phenotype_mapping = load_if_not_exists("loaded_data/sample_to_body_site_mapping.json", generate_phenotype_mapping)

  # Use phenotype mapping to create list of phenotypes for all the samples
  sample_body_site_phenotypes = load_if_not_exists("loaded_data/sample_body_site_phenotypes.npy", 
  get_phenotypes, 
  samples=samples, 
  sample_to_phenotype=phenotype_mapping)

  # Print number of "NA" phenotypes
  print("Number of 'NA' phenotypes: ", np.sum(sample_body_site_phenotypes == "NA"))

  data = load_if_not_exists("loaded_data/data.npy", calculate_data)

  print(data.shape , np.min(data) , np.max(data) ,np.mean(data))