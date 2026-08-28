import numpy as np

from preprocessing.artifact_paths import DEFAULT_CONFIG

"""
Compute mean and std for data normalization.
"""
data_path = DEFAULT_CONFIG.resized_expressions_path
X = np.load(data_path).astype(np.float32)   
print(f"  Shape: {X.shape}  Range: [{X.min():.4f}, {X.max():.4f}]")
print(f"  Mean: {X.mean():.4f}  Std: {X.std():.4f}")