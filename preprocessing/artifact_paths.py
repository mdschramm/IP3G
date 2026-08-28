"""
Single source of truth for preprocessing artifact paths and image-shape metadata.

Every consumer (preprocessing, gan, classifer, diffusion, evaluation) should import
paths/shapes from here instead of hardcoding "output/preprocessing" strings or
(width, height, channels) tuples locally.

Some artifacts depend only on the raw gene expression data and the t-SNE embedding —
they are identical regardless of target image size or channel count, so they always
live at SHARED_DIR (never duplicated per config):
    data.npy, tsne_results.npy, samples.npy, sample_to_body_site_mapping.json,
    sample_body_site_phenotypes.npy, primary_site_mapping.json,
    sample_primary_site_phenotypes.npy, gene_importance_order.npy, gene_f_stats.npy,
    y_primary_disease_or_tissue.npy, y_primary_site.npy

Other artifacts depend on the target (width, height, channels) — these live under a
PreprocessingConfig's artifact_dir, which resolves to SHARED_DIR itself for the
default (128, 128, 16) config (so every existing hardcoded path keeps working
unchanged) and to SHARED_DIR/"{W}x{H}x{C}" for any other config:
    unpadded_expressions.npy, resized_expressions.npy, pixel_occupancy_mask.npy,
    gene_pixel_channel.npy, channel_scales.npy, sigma_data.json

USAGE:
    from preprocessing.artifact_paths import DEFAULT_CONFIG, PreprocessingConfig

    DEFAULT_CONFIG.resized_expressions_path   # "output/preprocessing/resized_expressions.npy"
    PreprocessingConfig(256, 256, 1).resized_expressions_path
        # "output/preprocessing/256x256x1/resized_expressions.npy"
"""

import os
from dataclasses import dataclass

SHARED_DIR = "output/preprocessing"

DEFAULT_WIDTH = 128
DEFAULT_HEIGHT = 128
DEFAULT_CHANNELS = 16


@dataclass(frozen=True)
class PreprocessingConfig:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    channels: int = DEFAULT_CHANNELS

    @property
    def tag(self) -> str:
        return f"{self.width}x{self.height}x{self.channels}"

    @property
    def is_default(self) -> bool:
        return (self.width, self.height, self.channels) == (
            DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CHANNELS,
        )

    @property
    def artifact_dir(self) -> str:
        return SHARED_DIR if self.is_default else os.path.join(SHARED_DIR, self.tag)

    @property
    def evaluation_dir(self) -> str:
        base = "output/evaluation"
        return base if self.is_default else os.path.join(base, self.tag)

    @property
    def image_shape(self) -> tuple:
        return (self.width, self.height, self.channels)

    # ---- size/channel-DEPENDENT artifacts ----
    @property
    def unpadded_expressions_path(self) -> str:
        return os.path.join(self.artifact_dir, "unpadded_expressions.npy")

    @property
    def resized_expressions_path(self) -> str:
        return os.path.join(self.artifact_dir, "resized_expressions.npy")

    @property
    def pixel_occupancy_mask_path(self) -> str:
        return os.path.join(self.artifact_dir, "pixel_occupancy_mask.npy")

    @property
    def gene_pixel_channel_path(self) -> str:
        return os.path.join(self.artifact_dir, "gene_pixel_channel.npy")

    @property
    def channel_scales_path(self) -> str:
        return os.path.join(self.artifact_dir, "channel_scales.npy")

    @property
    def sigma_data_path(self) -> str:
        return os.path.join(self.artifact_dir, "sigma_data.json")


# ---- size/channel-INDEPENDENT artifacts — always under SHARED_DIR, never duplicated ----
DATA_PATH = os.path.join(SHARED_DIR, "data.npy")
TSNE_RESULTS_PATH = os.path.join(SHARED_DIR, "tsne_results.npy")
SAMPLES_PATH = os.path.join(SHARED_DIR, "samples.npy")
SAMPLE_BODY_SITE_MAPPING_PATH = os.path.join(SHARED_DIR, "sample_to_body_site_mapping.json")
SAMPLE_BODY_SITE_PHENOTYPES_PATH = os.path.join(SHARED_DIR, "sample_body_site_phenotypes.npy")
PRIMARY_SITE_MAPPING_PATH = os.path.join(SHARED_DIR, "primary_site_mapping.json")
SAMPLE_PRIMARY_SITE_PHENOTYPES_PATH = os.path.join(SHARED_DIR, "sample_primary_site_phenotypes.npy")
GENE_IMPORTANCE_ORDER_PATH = os.path.join(SHARED_DIR, "gene_importance_order.npy")
GENE_F_STATS_PATH = os.path.join(SHARED_DIR, "gene_f_stats.npy")
Y_PRIMARY_DISEASE_OR_TISSUE_PATH = os.path.join(SHARED_DIR, "y_primary_disease_or_tissue.npy")
Y_PRIMARY_SITE_PATH = os.path.join(SHARED_DIR, "y_primary_site.npy")

DEFAULT_CONFIG = PreprocessingConfig()
