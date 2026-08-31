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

# Which source corpus the artifacts were built from.
#
# Two distinct ideas, deliberately separated:
#
#   GTEX_DATASET   Structural. The original single-study corpus, whose paths stay
#                  exactly where they have always been. Every other dataset nests
#                  under a subdirectory. This is a property of the layout and must
#                  never be overridden — if it moved, a non-GTEx run would resolve
#                  to the unsuffixed paths and overwrite the GTEx artifacts.
#
#   DEFAULT_DATASET  Which corpus *this process* works on by default, selected with
#                  the RUN_DATASET env var. Mirrors how RUN_MODE selects local vs
#                  remote output directories, so remote runs need no new CLI flags.
# `or` rather than a get() default: an env var that is set but empty (which
# `docker run -e RUN_DATASET=` produces) must fall back, not resolve to "".
GTEX_DATASET = "gtex"
DEFAULT_DATASET = os.environ.get("RUN_DATASET") or GTEX_DATASET


@dataclass(frozen=True)
class PreprocessingConfig:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    channels: int = DEFAULT_CHANNELS
    dataset: str = DEFAULT_DATASET

    @property
    def tag(self) -> str:
        return f"{self.width}x{self.height}x{self.channels}"

    @property
    def is_default(self) -> bool:
        """True when the target geometry is the production 128x128x16 shape.

        Deliberately independent of `dataset`: it controls whether the geometry
        tag is appended, not which corpus root is used.
        """
        return (self.width, self.height, self.channels) == (
            DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CHANNELS,
        )

    @property
    def dataset_dir(self) -> str:
        """Root for this corpus's artifacts, size-dependent or not.

        Compares against GTEX_DATASET, never DEFAULT_DATASET: the unsuffixed path
        belongs to GTEx permanently, regardless of what RUN_DATASET is set to.
        """
        if self.dataset == GTEX_DATASET:
            return SHARED_DIR
        return os.path.join(SHARED_DIR, self.dataset)

    @property
    def artifact_dir(self) -> str:
        return self.dataset_dir if self.is_default else os.path.join(self.dataset_dir, self.tag)

    @property
    def evaluation_dir(self) -> str:
        base = "output/evaluation"
        if self.dataset != GTEX_DATASET:
            base = os.path.join(base, self.dataset)
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

    # ---- size/channel-INDEPENDENT artifacts — under dataset_dir, never duplicated ----
    @property
    def data_path(self) -> str:
        return os.path.join(self.dataset_dir, "data.npy")

    @property
    def tsne_results_path(self) -> str:
        return os.path.join(self.dataset_dir, "tsne_results.npy")

    @property
    def samples_path(self) -> str:
        return os.path.join(self.dataset_dir, "samples.npy")

    @property
    def gene_importance_order_path(self) -> str:
        return os.path.join(self.dataset_dir, "gene_importance_order.npy")

    @property
    def gene_f_stats_path(self) -> str:
        return os.path.join(self.dataset_dir, "gene_f_stats.npy")

    @property
    def gene_symbols_path(self) -> str:
        """Hugo symbols, row-aligned with data.npy's gene axis."""
        return os.path.join(self.dataset_dir, "gene_symbols.npy")

    @property
    def gene_entrez_path(self) -> str:
        return os.path.join(self.dataset_dir, "gene_entrez.npy")

    # ---- multi-attribute labels (rnaseqdb) ----
    @property
    def labels_path(self) -> str:
        """Per-sample label frame: sample_id + one column per attribute."""
        return os.path.join(self.dataset_dir, "labels.csv")

    @property
    def attribute_vocab_path(self) -> str:
        """Ordered value list per attribute, so index -> label is recoverable."""
        return os.path.join(self.dataset_dir, "attribute_vocab.json")

    def y_attribute_path(self, attribute: str) -> str:
        """One-hot labels for a single attribute, e.g. y_tissue.npy."""
        return os.path.join(self.dataset_dir, f"y_{attribute}.npy")


def model_output_dir(module: str, run_mode: str = None, dataset: str = None) -> str:
    """Output directory for a trained model's artifacts.

    Applies the same nesting rule as the preprocessing paths, one level deeper:

        output/classifier/remote            <- gtex  (unchanged, as it always was)
        output/classifier/remote/rnaseqdb   <- any other corpus

    Nesting *inside* the existing per-module directory is deliberate — the VM
    bind-mounts in gcloud_helpers (-v $VM_OUTPUT_BASE/classifier/remote:...) and
    the recursive gsutil rsync in sync_outputs both pick the subdirectory up for
    free, so adding a corpus needs no new mounts and no sync changes.

    Without this, a GTEx run and an RNAseqDB run would both write to
    output/{module}/remote and silently overwrite each other.

    Args:
        module: "classifier", "gan", or "diffusion"
        run_mode: defaults to $RUN_MODE, else "local"
        dataset: defaults to $RUN_DATASET, else "gtex"
    """
    run_mode = run_mode or os.environ.get("RUN_MODE", "local")
    dataset = dataset or DEFAULT_DATASET
    base = os.path.join("output", module, run_mode)
    return base if dataset == GTEX_DATASET else os.path.join(base, dataset)


DEFAULT_CONFIG = PreprocessingConfig()
RNASEQDB_CONFIG = PreprocessingConfig(dataset="rnaseqdb")

# ---- module-level aliases for the default (GTEx) corpus ----
# Kept so existing `from preprocessing.artifact_paths import DATA_PATH` imports
# keep resolving to exactly the paths they always have.
DATA_PATH = DEFAULT_CONFIG.data_path
TSNE_RESULTS_PATH = DEFAULT_CONFIG.tsne_results_path
SAMPLES_PATH = DEFAULT_CONFIG.samples_path
SAMPLE_BODY_SITE_MAPPING_PATH = os.path.join(SHARED_DIR, "sample_to_body_site_mapping.json")
SAMPLE_BODY_SITE_PHENOTYPES_PATH = os.path.join(SHARED_DIR, "sample_body_site_phenotypes.npy")
PRIMARY_SITE_MAPPING_PATH = os.path.join(SHARED_DIR, "primary_site_mapping.json")
SAMPLE_PRIMARY_SITE_PHENOTYPES_PATH = os.path.join(SHARED_DIR, "sample_primary_site_phenotypes.npy")
GENE_IMPORTANCE_ORDER_PATH = DEFAULT_CONFIG.gene_importance_order_path
GENE_F_STATS_PATH = DEFAULT_CONFIG.gene_f_stats_path
Y_PRIMARY_DISEASE_OR_TISSUE_PATH = os.path.join(SHARED_DIR, "y_primary_disease_or_tissue.npy")
Y_PRIMARY_SITE_PATH = os.path.join(SHARED_DIR, "y_primary_site.npy")
