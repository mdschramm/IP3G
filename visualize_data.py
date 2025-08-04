import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import ConvexHull

from preprocess_data import load_if_not_exists, calculate_data

def get_tsne_data(data):
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(data.T)
    return tsne_results

def plot_tsne(tsne_results):
    framedata = {"tsne-2d-one": tsne_results[:,0] , "tsne-2d-two": tsne_results[:,1] } 
    df = pd.DataFrame(framedata)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
            data=df,
            legend="full",
            alpha=0.3
        )
    plt.show()

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

    return rval
    

def plot_convex_hull(tsne_results, hull):
    plt.plot(tsne_results[:,0], tsne_results[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(tsne_results[simplex, 0], tsne_results[simplex, 1], 'k-')
    plt.show()

def plot_bounding_box(tsne_results, bbox):
    plt.scatter(tsne_results[:,0], tsne_results[:,1])
    plt.fill(bbox[:,0], bbox[:,1], alpha=0.2)
    plt.axis('equal')
    plt.show()

def rotate(p, origin=(0, 0), theta=0):

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def compute_rotation(bbox):
    theta = np.arctan((bbox[0][1]-bbox[1][1])/(bbox[0][0] - bbox[1][0]))
    return -theta

def initialize_image_data(sample_gene_expressions, normalized_tsne):
    w , h = np.max(normalized_tsne, axis=0)
    w, h = int(w) , int(h)
    data = np.zeros((sample_gene_expressions.shape[0], w+1, h+1))
    return data, w, h

def create_expression_images_from_tsne(sample_gene_expressions, normalized_tsne, data, w, h):
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
    return data

def pad_data(data, pad_size):
    left_padd = math.floor((pad_size - data.shape[1])/2)
    right_padd = math.ceil((pad_size - data.shape[1])/2)
    top_padd = math.floor((pad_size - data.shape[2])/2)
    bottom_padd = math.ceil((pad_size - data.shape[2])/2)
    data = np.pad(data , [(0,0),(left_padd,right_padd),(top_padd,bottom_padd)], 'constant')
    return data

if __name__ == "__main__":
    sample_gene_expressions = load_if_not_exists("loaded_data/data.npy", calculate_data)
    tsne_results = load_if_not_exists("loaded_data/tsne_results.npy", 
    get_tsne_data, 
    data=sample_gene_expressions)
    
    # plot_tsne(tsne_results)
    bbox = minimum_bounding_rectangle(tsne_results)
    # plot_bounding_box(tsne_results, bbox)
    # hull = ConvexHull(tsne_results)
    # plot_convex_hull(tsne_results, hull)

    theta = compute_rotation(bbox)
    rotated_tsne = rotate(tsne_results, origin=bbox[0], theta=theta)

    # rotated_bbox = rotate(bbox, origin=bbox[0], theta=theta)
    # plot_bounding_box(rotated_tsne, rotated_bbox)

    normalized_tsne = rotated_tsne - np.min(rotated_tsne, axis=0)

    data, w, h = initialize_image_data(sample_gene_expressions, normalized_tsne)

    data = create_expression_images_from_tsne(sample_gene_expressions, normalized_tsne, data, w, h)

    # Pad images to 128x128
    pad_size = 128
    data = pad_data(data, pad_size)

    # Todos
    # Generate some sample images for a few patient samples
    # Get phenotypes and y_train values for those patients
    # Get dimensions and total counts
    # Confirm with professor.

    

    

