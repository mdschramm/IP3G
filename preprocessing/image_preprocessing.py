"""
Image Preprocessing Pipeline for Gene Expression Data

This module transforms gene expression data into 2D image representations using t-SNE.

TRANSFORMATION PIPELINE:
1. Apply t-SNE to reduce gene dimensions from N_genes to 2D coordinates
2. Find minimum bounding rectangle around t-SNE point cloud
3. Rotate coordinates to align with bounding box axes
4. Normalize coordinates to start at origin (0,0), then scale uniformly to fit within 127×127
5. Map gene expressions to 2D images using t-SNE coordinates as pixel locations
6. Pad images to consistent dimensions (128×128) — t-SNE coordinates are pre-scaled so
   padding is the only spatial step needed; no interpolation, pixel values stay exactly [0, 1]

KEY CONCEPT:
- Each gene gets a 2D coordinate from t-SNE (similar genes cluster together)
- For each sample, create an image where pixel intensity = gene expression
- Genes with similar expression patterns appear at nearby pixels
- Result: Gene expression vector → 2D image suitable for CNN classification

FUNCTIONS:
- get_tsne_data(): Apply t-SNE dimensionality reduction
- minimum_bounding_rectangle(): Find optimal bounding box
- rotate(): Rotate 2D points
- compute_rotation(): Calculate rotation angle
- initialize_image_data(): Create empty image array
- create_expression_images_from_tsne(): Core transformation (expression → image)
- pad_data(): Center-pad images to target size
- resize_images(): Resize images using interpolation
"""

import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from scipy.ndimage import zoom
import math

TARGET_SIZE = 128


def get_tsne_data(data):
    """
    Apply t-SNE dimensionality reduction to map genes to 2D coordinates.
    
    Args:
        data: Gene expression matrix, shape (N_samples, N_genes)
        
    Returns:
        np.ndarray: t-SNE coordinates for each gene, shape (N_genes, 2)
        Each row represents a gene's 2D position in t-SNE space.
        Similar genes will have nearby coordinates.
    """
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(data.T)
    print("Finished t-SNE")
    return tsne_results


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    print("Finished minimum bounding rectangle")

    return rval


def rotate(p, origin=(0, 0), theta=0):
    """
    Rotate 2D points around an origin by angle theta.
    
    Args:
        p: Points to rotate, shape (N, 2)
        origin: Center of rotation, tuple (x, y)
        theta: Rotation angle in radians
        
    Returns:
        np.ndarray: Rotated points, shape (N, 2)
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def compute_rotation(bbox):
    """
    Compute rotation angle to align bounding box with axes.
    
    Args:
        bbox: Bounding box corners, shape (4, 2)
        
    Returns:
        float: Rotation angle in radians to align box with x/y axes
    """
    theta = np.arctan((bbox[0][1]-bbox[1][1])/(bbox[0][0] - bbox[1][0]))
    return -theta


def initialize_image_data(sample_gene_expressions, normalized_tsne):
    """
    Initialize empty image array based on t-SNE coordinate bounds.
    
    Args:
        sample_gene_expressions: Gene expression matrix, shape (N_samples, N_genes)
        normalized_tsne: Normalized t-SNE coordinates, shape (N_genes, 2)
        
    Returns:
        tuple: (data, w, h)
            - data: Empty image array, shape (N_samples, w+1, h+1)
            - w: Width of image (max x-coordinate)
            - h: Height of image (max y-coordinate)
    """
    w , h = np.max(normalized_tsne, axis=0)
    w, h = int(w) , int(h)
    data = np.zeros((sample_gene_expressions.shape[0], w+1, h+1))
    return data, w, h


def create_expression_images_from_tsne(sample_gene_expressions, normalized_tsne, data, w, h):
    """
    Convert gene expression vectors to 2D images using t-SNE coordinates.
    
    For each sample:
    1. Map each gene's expression value to a pixel at its t-SNE coordinate
    2. If multiple genes map to same pixel, average their expression values
    3. Normalize pixel intensities to [0, 1] range
    
    Args:
        sample_gene_expressions: Gene expression matrix, shape (N_samples, N_genes)
        normalized_tsne: t-SNE coordinates for genes, shape (N_genes, 2)
        data: Pre-initialized image array, shape (N_samples, w+1, h+1)
        w: Image width
        h: Image height
        
    Returns:
        np.ndarray: Expression images, shape (N_samples, w+1, h+1)
        Pixel values in [0, 1] representing normalized gene expression.
    """
    # Go through each sample's gene expressions
    for i, profile in enumerate(sample_gene_expressions):
        counts = np.zeros((w+1,h+1))
        for j in range(sample_gene_expressions.shape[1]):
            # acquire pixel values for sample on tsne
            # (similar genes will be close together on tsne)
            x1 , x2 = int(normalized_tsne[j][0]) , int(normalized_tsne[j][1])
            # Set sample's pixel values to gene expression based on tsne location
            data[i,x1,x2] = data[i,x1,x2] + profile[j]
            counts[x1,x2] +=1
        # Normalize gene expression values for sample image
        for k in range(w+1):
            for l in range(h+1):
                if counts[k,l]>1 :
                    data[i , k , l] /= counts[k,l]

    # Normalize gene expression values for all samples
    data = data / np.max(data)
    print("Finished create expression images from tsne")
    return data


def pad_data(data, pad_size):
    """
    Center-pad images to target size with zeros.
    
    Args:
        data: Image array, shape (N_samples, w, h)
        pad_size: Target size for width and height
        
    Returns:
        np.ndarray: Padded images, shape (N_samples, pad_size, pad_size)
        If input is larger than pad_size, returns original data with warning.
    """
    if data.shape[1] > pad_size or data.shape[2] > pad_size:
        print(f"Warning: Data shape {data.shape[1]}x{data.shape[2]} is larger than pad size {pad_size}")
        return data
    left_padd = max(0, math.floor((pad_size - data.shape[1])/2))
    right_padd = max(0, math.ceil((pad_size - data.shape[1])/2))
    top_padd = max(0, math.floor((pad_size - data.shape[2])/2))
    bottom_padd = max(0, math.ceil((pad_size - data.shape[2])/2))
    data = np.pad(data , [(0,0),(left_padd,right_padd),(top_padd,bottom_padd)], 'constant')
    return data

# unused
def resize_images(images, target_size=TARGET_SIZE):
    """
    Resize images to target dimensions using cubic interpolation.
    
    Args:
        images: Image array, shape (N_samples, w, h)
        target_size: Target width and height (default: 128)
        
    Returns:
        np.ndarray: Resized images, shape (N_samples, target_size, target_size)
    """
    n, w, h = images.shape
    zoom_factors = (1, target_size / w, target_size / h)
    resized = zoom(images, zoom_factors, order=3)  # order=3: cubic interpolation
    return resized
