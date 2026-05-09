"""
Visualization Tools for Gene Expression Analysis

This module provides visualization functions for exploratory data analysis
of gene expression data and t-SNE transformations.

VISUALIZATION FUNCTIONS:
- plot_tsne(): Scatter plot of t-SNE results to visualize gene clustering
- plot_convex_hull(): Visualize convex hull around t-SNE point cloud
- plot_bounding_box(): Visualize minimum bounding rectangle
- render_image(): Display individual expression images

These functions are useful for:
- Debugging the preprocessing pipeline
- Understanding gene clustering patterns
- Validating t-SNE transformations
- Inspecting generated expression images
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_tsne(tsne_results, output_path="output/preprocessing/tsne_scatter.png"):
    """
    Visualize t-SNE results as a scatter plot and save to file.
    
    Args:
        tsne_results: t-SNE coordinates, shape (N_genes, 2)
        output_path: Path to save the plot (default: output/preprocessing/tsne_scatter.png)
        
    Returns:
        None (saves plot to file)
    """
    framedata = {"tsne-2d-one": tsne_results[:,0] , "tsne-2d-two": tsne_results[:,1] } 
    df = pd.DataFrame(framedata)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
            data=df,
            legend="full",
            alpha=0.3
        )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE scatter plot saved to {output_path}")


def plot_convex_hull(tsne_results, hull, output_path="output/preprocessing/tsne_convex_hull.png"):
    """
    Visualize convex hull around t-SNE points and save to file.
    
    Args:
        tsne_results: t-SNE coordinates, shape (N_genes, 2)
        hull: scipy.spatial.ConvexHull object
        output_path: Path to save the plot (default: output/preprocessing/tsne_convex_hull.png)
        
    Returns:
        None (saves plot to file)
    """
    plt.figure(figsize=(12,10))
    plt.plot(tsne_results[:,0], tsne_results[:,1], 'o', alpha=0.5)
    for simplex in hull.simplices:
        plt.plot(tsne_results[simplex, 0], tsne_results[simplex, 1], 'k-')
    plt.title('t-SNE Convex Hull')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convex hull plot saved to {output_path}")


def plot_bounding_box(tsne_results, bbox, output_path="output/preprocessing/tsne_bounding_box.png"):
    """
    Visualize minimum bounding rectangle around t-SNE points and save to file.
    
    Args:
        tsne_results: t-SNE coordinates, shape (N_genes, 2)
        bbox: Bounding box corners, shape (4, 2)
        output_path: Path to save the plot (default: output/preprocessing/tsne_bounding_box.png)
        
    Returns:
        None (saves plot to file)
    """
    plt.figure(figsize=(12,10))
    plt.scatter(tsne_results[:,0], tsne_results[:,1], alpha=0.5, s=10)
    plt.fill(bbox[:,0], bbox[:,1], alpha=0.2, color='red', edgecolor='red', linewidth=2)
    plt.title('t-SNE Minimum Bounding Rectangle')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.axis('equal')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bounding box plot saved to {output_path}")


def render_image(image, output_path="output/preprocessing/sample_expression_image.png", title="Expression Image"):
    """
    Save a single expression image to file.
    
    Args:
        image: 2D array representing an expression image, shape (w, h)
        output_path: Path to save the image (default: output/preprocessing/sample_expression_image.png)
        title: Title for the plot
        
    Returns:
        None (saves image to file)
    """
    plt.figure(figsize=(8,8))
    plt.imshow(image, cmap='viridis')
    plt.colorbar(label='Normalized Expression')
    plt.title(title)
    plt.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Expression image saved to {output_path}")
