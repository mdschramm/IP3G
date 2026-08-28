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
        data: Image array, shape (N_samples, w, h) or (N_samples, w, h, C)
        pad_size: Target size for width and height

    Returns:
        np.ndarray: Padded images, shape (N_samples, pad_size, pad_size) or (N_samples, pad_size, pad_size, C)
        If input is larger than pad_size, returns original data with warning.
    """
    if data.shape[1] > pad_size or data.shape[2] > pad_size:
        print(f"Warning: Data shape {data.shape[1]}x{data.shape[2]} is larger than pad size {pad_size}")
        return data
    left_padd = max(0, math.floor((pad_size - data.shape[1])/2))
    right_padd = max(0, math.ceil((pad_size - data.shape[1])/2))
    top_padd = max(0, math.floor((pad_size - data.shape[2])/2))
    bottom_padd = max(0, math.ceil((pad_size - data.shape[2])/2))
    if data.ndim == 4:
        data = np.pad(data, [(0,0),(left_padd,right_padd),(top_padd,bottom_padd),(0,0)], 'constant')
    else:
        data = np.pad(data, [(0,0),(left_padd,right_padd),(top_padd,bottom_padd)], 'constant')
    return data


def compute_gene_importance_order(sample_gene_expressions, phenotypes):
    """
    Rank genes by ANOVA F-statistic across tissue classes (descending).

    Genes with higher F-statistics have stronger class-discriminative signal and
    are assigned to lower channel indices in the multichannel image, where they
    receive dedicated 1-to-1 pixel slots rather than being averaged into overflow.

    Args:
        sample_gene_expressions: (N_samples, N_genes) float32
        phenotypes: (N_samples,) string labels (body site)

    Returns:
        gene_importance_order: (N_genes,) int64, indices of genes sorted by F-stat descending
        f_stats: (N_genes,) float32, F-statistic per gene
    """
    X = sample_gene_expressions.astype(np.float64)
    N, G = X.shape
    labels_unique = sorted(set(phenotypes))
    K = len(labels_unique)
    label_to_idx = {l: i for i, l in enumerate(labels_unique)}
    labels = np.array([label_to_idx[p] for p in phenotypes])

    grand_mean = X.mean(axis=0)                          # (G,)
    class_means = np.zeros((K, G), dtype=np.float64)
    class_counts = np.zeros(K, dtype=np.float64)

    for k in range(K):
        mask = labels == k
        class_means[k] = X[mask].mean(axis=0)
        class_counts[k] = mask.sum()

    ss_between = np.sum(class_counts[:, None] * (class_means - grand_mean) ** 2, axis=0)
    ss_within = sum(
        np.sum((X[labels == k] - class_means[k]) ** 2, axis=0) for k in range(K)
    )
    f_stats = (ss_between / (K - 1)) / (ss_within / (N - K) + 1e-10)
    f_stats = f_stats.astype(np.float32)

    gene_importance_order = np.argsort(f_stats)[::-1].astype(np.int64)
    print(f"  Gene importance order computed. Top F-stat: {f_stats[gene_importance_order[0]]:.1f}  "
          f"Median: {np.median(f_stats):.1f}")
    return gene_importance_order, f_stats


def create_multichannel_expression_images_from_tsne(
    sample_gene_expressions, normalized_tsne, gene_importance_order, w, h, n_channels=16
):
    """
    Convert gene expression vectors to 16-channel 2D images using t-SNE coordinates.

    Genes are assigned to channels in descending F-statistic order: the most
    discriminative gene at each pixel goes into channel 0, the next into channel 1,
    etc.  Channels 0-14 each hold exactly one gene per pixel; channel 15 holds the
    mean of all overflow genes (those beyond the 14th at a given pixel).

    Forward normalization (3 steps):
      1. Channel assignment: fill_count tracks next available channel per pixel.
         Genes 0-14 at a pixel get a dedicated slot; gene 15+ accumulates in ch15.
      2. Averaging: channel-15 pixels with >1 overflow gene are divided by overflow count.
      3. Per-channel normalization: each channel divided by its own max value so that
         all channels are independently in [0, 1]. channel_scales[k] stores the pre-
         division max and is used by the reverse mapping to recover RSEM units.

    Args:
        sample_gene_expressions: (N_samples, N_genes) float32 RSEM counts
        normalized_tsne: (N_genes, 2) float array (already scaled to 0..TARGET_SIZE-1)
        gene_importance_order: (N_genes,) int64, genes sorted by F-stat descending
        w: image width  (max x-coordinate, inclusive)
        h: image height (max y-coordinate, inclusive)
        n_channels: number of channels (default 16)

    Returns:
        data: (N_samples, w+1, h+1, n_channels) float32 in [0, 1]
        pixel_occupancy_mask: (w+1, h+1, n_channels) bool, True where a gene is assigned
        gene_pixel_channel: (N_genes, 3) int32  [px, py, ch] per gene, original ordering
        channel_scales: (n_channels,) float32 pre-division channel maxima
    """
    N_samples, N_genes = sample_gene_expressions.shape
    data = np.zeros((N_samples, w + 1, h + 1, n_channels), dtype=np.float32)

    # Build pixel→channel assignment from gene importance order (same for all samples)
    fill_count = np.zeros((w + 1, h + 1), dtype=np.int32)
    overflow_count = np.zeros((w + 1, h + 1), dtype=np.int32)

    # gene_pixel_channel[g] = [px, py, ch] in the ORIGINAL gene ordering
    gene_pixel_channel = np.zeros((N_genes, 3), dtype=np.int32)
    # Track mapping from importance-sorted index → original gene index
    for rank, gene_idx in enumerate(gene_importance_order):
        px = int(normalized_tsne[gene_idx][0])
        py = int(normalized_tsne[gene_idx][1])
        c = fill_count[px, py]
        if c < n_channels - 1:
            gene_pixel_channel[gene_idx] = [px, py, c]
            fill_count[px, py] += 1
        else:
            gene_pixel_channel[gene_idx] = [px, py, n_channels - 1]
            overflow_count[px, py] += 1

    # occupancy mask: True wherever at least one gene is assigned
    pixel_occupancy_mask = np.zeros((w + 1, h + 1, n_channels), dtype=bool)
    for gene_idx in range(N_genes):
        px, py, ch = gene_pixel_channel[gene_idx]
        pixel_occupancy_mask[px, py, ch] = True

    print(f"  Pixel-channel assignment done. "
          f"Overflow pixels (ch{n_channels - 1}): {np.sum(overflow_count > 0):,}  "
          f"Total occupied positions: {np.sum(pixel_occupancy_mask):,}")

    # Fill data for each sample
    for i, profile in enumerate(sample_gene_expressions):
        if i % 1000 == 0:
            print(f"  Creating multichannel images: sample {i}/{N_samples}")
        for gene_idx in range(N_genes):
            px, py, ch = gene_pixel_channel[gene_idx]
            data[i, px, py, ch] += profile[gene_idx]

        # Average channel-15 pixels that received >1 overflow gene
        ov_mask = overflow_count > 1
        data[i, :, :, n_channels - 1][ov_mask] /= overflow_count[ov_mask]

    # Per-channel normalization — each channel independently to [0, 1]
    channel_scales = np.zeros(n_channels, dtype=np.float32)
    for k in range(n_channels):
        ch_max = float(np.max(data[:, :, :, k]))
        if ch_max > 0:
            channel_scales[k] = ch_max
            data[:, :, :, k] /= ch_max
        else:
            channel_scales[k] = 1.0

    print(f"  Multichannel images created. Shape: {data.shape}  "
          f"Value range: [{data.min():.4f}, {data.max():.4f}]")
    return data, pixel_occupancy_mask, gene_pixel_channel, channel_scales

# unused
# def resize_images(images, target_size=TARGET_SIZE):
#     """
#     Resize images to target dimensions using cubic interpolation.
    
#     Args:
#         images: Image array, shape (N_samples, w, h)
#         target_size: Target width and height (default: 128)
        
#     Returns:
#         np.ndarray: Resized images, shape (N_samples, target_size, target_size)
#     """
#     n, w, h = images.shape
#     zoom_factors = (1, target_size / w, target_size / h)
#     resized = zoom(images, zoom_factors, order=3)  # order=3: cubic interpolation
#     return resized
