# Refactoring Suggestions: Separate Visualization from Preprocessing

## Current Issue
`visualize_data.py` mixes two concerns:
1. **Data preprocessing** (t-SNE transformation, image creation, label encoding)
2. **Visualization** (plotting functions for exploratory analysis)

## Proposed Refactoring

### 1. Create `image_preprocessing.py`
Move core preprocessing functions that transform data (not visualization):

**Functions to move:**
- `get_tsne_data()` - t-SNE transformation
- `minimum_bounding_rectangle()` - Geometric computation
- `rotate()` - Coordinate transformation
- `compute_rotation()` - Geometric computation
- `initialize_image_data()` - Data structure creation
- `create_expression_images_from_tsne()` - Core transformation
- `pad_data()` - Image preprocessing
- `resize_images()` - Image preprocessing
- `get_y_train()` - Label encoding

**Rationale:** These functions perform data transformations needed for model training, not visualization.

### 2. Keep in `visualize_data.py` (rename to `visualization.py`)
Visualization-only functions:

**Functions to keep:**
- `plot_tsne()` - Visualization
- `plot_convex_hull()` - Visualization
- `plot_bounding_box()` - Visualization
- `render_image()` - Visualization

**Rationale:** These are purely for exploratory data analysis and debugging.

### 3. Update `preprocess_data.py`
Add the label encoding function since it's a core preprocessing step:

**Function to add:**
- `get_y_train()` - Converts phenotypes to one-hot encoded labels

**Rationale:** Label encoding is a fundamental preprocessing step, similar to other functions already in this module.

## Proposed File Structure

```
IP3G/
├── preprocess_data.py          # Raw data loading & basic preprocessing
│   ├── load_samples()
│   ├── calculate_data()
│   ├── generate_phenotype_mapping()
│   ├── get_phenotypes()
│   └── get_y_train()          # NEW: Move from visualize_data.py
│
├── image_preprocessing.py      # NEW: t-SNE & image transformation
│   ├── get_tsne_data()
│   ├── minimum_bounding_rectangle()
│   ├── rotate()
│   ├── compute_rotation()
│   ├── initialize_image_data()
│   ├── create_expression_images_from_tsne()
│   ├── pad_data()
│   └── resize_images()
│
├── visualization.py            # Renamed from visualize_data.py
│   ├── plot_tsne()
│   ├── plot_convex_hull()
│   ├── plot_bounding_box()
│   └── render_image()
│
└── Classifier.py               # Model training (unchanged)
```

## Migration Steps

1. **Create `image_preprocessing.py`**
   - Copy preprocessing functions from `visualize_data.py`
   - Add module docstring explaining t-SNE → image pipeline
   - Update imports

2. **Update `preprocess_data.py`**
   - Add `get_y_train()` function
   - Add `to_categorical` import from keras

3. **Rename and clean `visualize_data.py` → `visualization.py`**
   - Keep only visualization functions
   - Update imports to reference `image_preprocessing.py`
   - Simplify module docstring to focus on visualization

4. **Update `Classifier.py`**
   - Update imports to use new module structure
   - No functional changes needed

5. **Update main execution scripts**
   - The `if __name__ == "__main__"` block in current `visualize_data.py` should move to a separate script like `prepare_training_data.py`

## Benefits

1. **Separation of Concerns**: Preprocessing vs. visualization clearly separated
2. **Reusability**: Can import preprocessing functions without matplotlib dependencies
3. **Testing**: Easier to test preprocessing pipeline independently
4. **Clarity**: Each module has a single, clear purpose
5. **Performance**: Can skip visualization imports when running headless (e.g., on cloud VMs)

## Example: New `prepare_training_data.py`

```python
"""
Main script to prepare training data from raw gene expression files.
Orchestrates the full preprocessing pipeline.
"""
from preprocess_data import load_if_not_exists, calculate_data, load_samples, generate_phenotype_mapping, get_phenotypes, get_y_train, GTEX_PHENOTYPE
from image_preprocessing import get_tsne_data, minimum_bounding_rectangle, rotate, compute_rotation, initialize_image_data, create_expression_images_from_tsne, pad_data, resize_images

TARGET_SIZE = 128

if __name__ == "__main__":
    # Load gene expressions
    sample_gene_expressions = load_if_not_exists("loaded_data/data.npy", calculate_data)
    
    # Apply t-SNE
    tsne_results = load_if_not_exists("loaded_data/tsne_results.npy", 
        get_tsne_data, data=sample_gene_expressions)
    
    # Transform to images
    bbox = minimum_bounding_rectangle(tsne_results)
    theta = compute_rotation(bbox)
    rotated_tsne = rotate(tsne_results, origin=bbox[0], theta=theta)
    normalized_tsne = rotated_tsne - np.min(rotated_tsne, axis=0)
    
    data, w, h = initialize_image_data(sample_gene_expressions, normalized_tsne)
    data = load_if_not_exists("loaded_data/unpadded_expressions.npy", 
        create_expression_images_from_tsne, 
        sample_gene_expressions=sample_gene_expressions,
        normalized_tsne=normalized_tsne, data=data, w=w, h=h)
    
    data = pad_data(data, TARGET_SIZE)
    data = load_if_not_exists("loaded_data/resized_expressions.npy",
        resize_images, images=data, target_size=TARGET_SIZE)
    
    # Prepare labels
    samples = load_if_not_exists("loaded_data/samples.npy", load_samples)
    phenotype_mapping = load_if_not_exists("loaded_data/sample_to_body_site_mapping.json", 
        generate_phenotype_mapping)
    sample_body_site_phenotypes = load_if_not_exists("loaded_data/sample_body_site_phenotypes.npy", 
        get_phenotypes, samples=samples, sample_to_phenotype=phenotype_mapping)
    y_train = load_if_not_exists("loaded_data/y_primary_disease_or_tissue.npy",
        get_y_train, phenotypes=sample_body_site_phenotypes)
    
    print(f"Training data prepared: {data.shape}, Labels: {y_train.shape}")
```

## Optional: Further Improvements

1. **Create a `Pipeline` class** to encapsulate the entire preprocessing flow
2. **Add configuration file** (YAML/JSON) for hyperparameters like `TARGET_SIZE`
3. **Add data validation** to check shape consistency between steps
4. **Add logging** instead of print statements for better debugging
