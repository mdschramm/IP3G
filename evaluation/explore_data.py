
import numpy as np
import matplotlib.pyplot as plt
import os

from preprocessing.artifact_paths import DEFAULT_CONFIG, Y_PRIMARY_DISEASE_OR_TISSUE_PATH

def get_class_stats():
    y = np.load(Y_PRIMARY_DISEASE_OR_TISSUE_PATH)
    # convert from 1hot encoding
    y = np.argwhere(y == 1)[:, 1]
    print(f'Total samples: {len(y)}')
    n_classes = len(set(np.unique(y)))
    print(f'Num classes: {n_classes}')
    unique, counts = np.unique(y.astype(int), return_counts=True)
    print(f'\nClass | Count | % ')
    print('-' * 30)
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f'  {cls:3d}  |  {cnt:4d}  | {cnt/len(y)*100:.1f}%')
    print(f'\nMin class size: {counts.min()}')
    print(f'Max class size: {counts.max()}')
    print(f'Mean class size: {counts.mean():.1f}')
    print(f'Median class size: {np.median(counts):.1f}')

    """
    Findings: Should remove low frequency < 20 classes
    Total samples: 7851
    Num classes: 54

    Class | Count | % 
    ------------------------------
    38  |   396  | 5.0%
    53  |   337  | 4.3%
    45  |   324  | 4.1%
        0  |   318  | 4.1%
    36  |   288  | 3.7%
        5  |   281  | 3.6%
    50  |   279  | 3.6%
    39  |   278  | 3.5%
    29  |   273  | 3.5%
    23  |   256  | 3.3%
    30  |   245  | 3.1%
    44  |   233  | 3.0%
        3  |   207  | 2.6%
    33  |   202  | 2.6%
        1  |   197  | 2.5%
    20  |   179  | 2.3%
    32  |   175  | 2.2%
    48  |   175  | 2.2%
    27  |   167  | 2.1%
    41  |   167  | 2.1%
    49  |   165  | 2.1%
    26  |   141  | 1.8%
    28  |   137  | 1.7%
        2  |   128  | 1.6%
    11  |   119  | 1.5%
        4  |   118  | 1.5%
    35  |   110  | 1.4%
        9  |   109  | 1.4%
    21  |   107  | 1.4%
    42  |   107  | 1.4%
    12  |   105  | 1.3%
    16  |   104  | 1.3%
    43  |   100  | 1.3%
    47  |   100  | 1.3%
    13  |    95  | 1.2%
    10  |    93  | 1.2%
    46  |    92  | 1.2%
    40  |    88  | 1.1%
    52  |    85  | 1.1%
    14  |    84  | 1.1%
        8  |    83  | 1.1%
    15  |    82  | 1.0%
    17  |    81  | 1.0%
    51  |    78  | 1.0%
    22  |    70  | 0.9%
        7  |    69  | 0.9%
    18  |    60  | 0.8%
    19  |    57  | 0.7%
    37  |    55  | 0.7%
    34  |    28  | 0.4%
        6  |     9  | 0.1%
    24  |     6  | 0.1%
    31  |     5  | 0.1%
    25  |     4  | 0.1%

    Min class size: 4
    Max class size: 396
    Mean class size: 145.4
    Median class size: 109.5
    """

def show_example_images_from_each_class(n=4, selected_classes=None):
    # Create evaluation/explore if doesn't already exist
    if not os.path.exists('evaluation/explore'):
        os.makedirs('evaluation/explore')
    x = np.load(DEFAULT_CONFIG.resized_expressions_path)
    y = np.load(Y_PRIMARY_DISEASE_OR_TISSUE_PATH)
    # convert from 1hot encoding
    y = np.argwhere(y == 1)[:, 1]
    unique = np.unique(y)
    excluded_classes = [6, 24, 25, 31]
    for cls in unique:
        if cls in excluded_classes: continue
        if selected_classes is not None and cls not in selected_classes: continue
        idx = np.where(y == cls)[0]
        idx = np.random.choice(idx, n, replace=False)
        # Group classes into multi-plot
        n_cols = (n + 1) // 2
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))
        for i in range(n):
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(x[idx[i]], cmap='viridis')
            ax.set_title(f"Class {cls}")
        for i in range(n, 2 * n_cols):
            axes[i // n_cols, i % n_cols].axis('off')
        plt.savefig(f"evaluation/explore/examples_class_{cls}.png")

show_example_images_from_each_class(selected_classes=[0, 23, 38, 53])
 
def show_pixel_distribution():
    """
    1. Transform 
    """
    # X_train = np.load(DEFAULT_CONFIG.resized_expressions_path).astype(np.float32)
    X_train = np.load(DEFAULT_CONFIG.resized_expressions_path).astype(np.float32)
    plt.hist(X_train.flatten(), bins=100)
    plt.savefig('evaluation/explore/pixel_distribution.png')

# show_pixel_distribution()

def show_stats():
    X_train = np.load(DEFAULT_CONFIG.resized_expressions_path).astype(np.float32)
    # Shape
    print(f"Shape: {X_train.shape}")
    # Range
    print(f"Range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    # Mean
    print(f"Mean: {X_train.mean():.4f}")
    # Std
    print(f"Std: {X_train.std():.4f}")
    # Number of 0 valued pixels
    num_pixels = (X_train == 0).sum()
    print(f"Number of 0 valued pixels: {num_pixels}")
    # Number of pixels below 0.5
    num_pixels = (X_train < 0.5).sum()
    print(f"Number of pixels below 0.5: {num_pixels}")
    # Number of pixels above 0.5
    num_pixels = (X_train > 0.5).sum()
    print(f"Number of pixels above 0.5: {num_pixels}")

show_stats()
# get_class_stats()