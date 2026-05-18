import numpy as np

EXCLUDED_CLASSES = [6, 24, 25, 31]


def filter_classes(X, y_onehot, excluded_classes=None):
    """Remove samples whose class is in excluded_classes.

    Args:
        X: ndarray [N, ...]
        y_onehot: one-hot encoded labels [N, num_classes]
        excluded_classes: list of integer class indices to drop (default: EXCLUDED_CLASSES)

    Returns:
        X_filtered, y_onehot_filtered
    """
    if excluded_classes is None:
        excluded_classes = EXCLUDED_CLASSES
    if not excluded_classes:
        return X, y_onehot
    class_indices = np.argmax(y_onehot, axis=1)
    mask = ~np.isin(class_indices, excluded_classes)
    return X[mask], y_onehot[mask]
