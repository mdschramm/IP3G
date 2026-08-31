"""
Configuration for EDM2 Conditional Diffusion with Classifier-Free Guidance.

Provides separate configurations for local (Mac M2) and remote (A100) training.
Noise parameterization uses EDM2 (Karras et al. 2022/2023): continuous sigma
sampled from a log-normal, with c_skip/c_out/c_in preconditioning so the loss
weight w(σ)·c_out²=1 at all noise levels.
"""

import json
import math
import os

from preprocessing.artifact_paths import DEFAULT_CONFIG, GTEX_DATASET

# Local configuration - Mac M2, 16GB RAM (architectural sanity check, ~2-3 hours)
CONFIG_LOCAL = {
    # Model architecture
    'image_size': DEFAULT_CONFIG.width,
    'in_channels': DEFAULT_CONFIG.channels,
    'channels': [32, 64, 128],           # 3 levels (kept small for M2)
    'num_res_blocks': 2,                  # Per resolution
    'attention_resolutions': [],          # No mid-run attention on M2 (bottleneck only)
    'num_heads': 4,
    'dropout': 0.1,
    'embedding_dim': 256,                 # Time + class embeddings
    'num_classes': 54,                    # GTEx tissue/disease classes
    'excluded_classes': [6, 24, 25, 31],  # Classes with <10 samples — too few for CFG
    'use_sparse_attention': False,
    'sparse_top_k_frac': 0.5,
    'res_balance': 0.3,                   # mp_sum weight: (1-t)*skip + t*residual; 0.3 → 70/30 (EDM2 default)

    # Training (~2-3 hours on M2)
    'batch_size': 8,                      # M1/M2 memory headroom
    'learning_rate': 3e-4,
    'lr_schedule': 'flat',                # warmup + constant; plateau = model, not LR→0
    'num_steps': 1000,
    'save_interval': 999_999,
    'sample_interval': 250,
    'log_interval': 1,
    'diag_interval': 20,
    'ema_decay': 0.99,                    # Lower so EMA tracks fast in short runs
    'gradient_clip': 1.0,
    'warmup_steps': 50,
    'mixed_precision': True,

    # EDM2 noise parameterization
    'sigma_data': 0.1709, # (from occupied pixel variance shown in sigma_data.json)                  # std of OCCUPIED positions only (measured: output/preprocessing/sigma_data.json → "occupied"); update after re-running preprocessing
    'P_mean': -1.77,                       # ln(sigma_data_occupied); update to match after re-running preprocessing
    'P_std': 1.2,                         # log-normal σ spread (paper default)
    'sigma_min': 0.002,                   # near-clean inference endpoint
    'sigma_max': 80.0,                    # pure-noise inference starting point
    'sigma_rho': 7,                       # Karras step-density exponent (more steps near sigma_min)

    # Classifier-free guidance
    'dropout_rate': 0.10,

    # FP16 safety
    'act_clip_magnitude': 256.0,          # Clamp encoder/decoder block outputs to ±this value (EDM2 §B)

    # Logvar MLP (EDM2 §B.3): Fourier feature channels for the noise-level uncertainty head
    'logvar_channels': 128,

    # Data
    'data_dir': DEFAULT_CONFIG.artifact_dir,
    'feature_file': 'resized_expressions.npy',
    'label_file': 'y_primary_disease_or_tissue.npy',
    'checkpoint_dir': 'output/diffusion/local/checkpoints',
    'sample_dir': 'output/diffusion/local/samples',
}

# Diagnostic configuration — the actual config used for remote (A100) runs.
# Intent: validate model behaviour (loss shape, CFG separation, gradient norms)
# on the full-size architecture without committing to an unbounded run.
CONFIG_DIAGNOSTIC = {
    # Architecture — identical to remote so results are representative
    'image_size': DEFAULT_CONFIG.width,
    'in_channels': DEFAULT_CONFIG.channels,
    'channels': [32, 64, 128, 256],
    'num_res_blocks': 3,
    'attention_resolutions': [16],
    'num_heads': 4,
    'dropout': 0.1,
    'embedding_dim': 256,
    'num_classes': 54,
    'excluded_classes': [6, 24, 25, 31],
    'use_sparse_attention': False,
    'sparse_top_k_frac': 0.5,
    'res_balance': 0.3,

    # Short training run
    'batch_size': 32,
    'learning_rate': 3e-4,
    'lr_schedule': 'cosine',
    'num_steps': 30_000,
    'save_interval': 999_999,            # Only the final save fires
    'sample_interval': 2_000,
    'log_interval': 100,
    'diag_interval': 500,
    'ema_decay': 0.9995,
    'gradient_clip': 1.0,
    'warmup_steps': 1500,                # ~5% of num_steps
    'mixed_precision': True,

    # EDM2 noise parameterization
    'sigma_data': 0.1709, # (from occupied pixel variance shown in sigma_data.json)
    'P_mean': -1.77,
    'P_std': 1.2,
    'sigma_min': 0.002,
    'sigma_max': 80.0,
    'sigma_rho': 7,

    # CFG
    'dropout_rate': 0.1,

    # FP16 safety
    'act_clip_magnitude': 256.0,

    # Logvar MLP
    'logvar_channels': 128,

    # Data
    'data_dir': DEFAULT_CONFIG.artifact_dir,
    'feature_file': 'resized_expressions.npy',
    'label_file': 'y_primary_disease_or_tissue.npy',
    'checkpoint_dir': 'output/diffusion/diagnostic/checkpoints',
    'sample_dir': 'output/diffusion/diagnostic/samples',
}


# Order is load-bearing: column a of class_labels indexes embedding table a, so
# this tuple defines the contract between the dataset, the model and the sampler.
#
# `source` is included deliberately even though it is near-collinear with
# `condition` (GTEx is 100% normal). Conditioning on it costs one small table and
# buys the ability to ASK for a TCGA-style normal vs a GTEx-style normal, which is
# exactly the axis a residual batch effect would live on. If ComBat fully removed
# the batch effect, guidance along source does nothing and the table decays to
# noise — an informative null result either way.
#
# Note that `subtype` carries a real "none" code AND a null token, and they mean
# different things: "none" asserts the sample is a normal with no TCGA code, the
# null token asserts nothing at all about subtype.
FACTORIZED_ATTRIBUTES = ('tissue', 'condition', 'subtype', 'source')


def _factorized_overlay(config):
    """Rewrite a GTEx config in place for the RNAseqDB corpus.

    Selected by RUN_DATASET, never by a flag — DEFAULT_CONFIG has already
    re-pointed data_dir at output/preprocessing/rnaseqdb/, so this only has to
    fix the parts of the config that are corpus-specific: the label contract,
    the measured sigma_data, and the output directories.
    """
    vocab_path = DEFAULT_CONFIG.attribute_vocab_path
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"RUN_DATASET={DEFAULT_CONFIG.dataset} needs {vocab_path}.\n"
            "Build it first:  RUN_DATASET=rnaseqdb python -m preprocessing.prepare_rnaseqdb_data"
        )
    with open(vocab_path) as fh:
        vocab = json.load(fh)

    config['attributes'] = [(name, len(vocab[name])) for name in FACTORIZED_ATTRIBUTES]
    # num_classes is meaningless once conditioning is factorized; keep it as the
    # tissue count so anything that still reads it degrades sensibly rather than
    # indexing a 54-class GTEx vocabulary that does not exist here.
    config['num_classes'] = len(vocab['tissue'])
    config['label_file'] = None                 # labels come from y_<attr>.npy, stacked
    config['excluded_classes'] = []             # every RNAseqDB class is retained

    # sigma_data is a MEASURED property of the corpus, not a hyperparameter. Using
    # the GTEx 0.1709 here would mis-set the entire EDM2 preconditioning schedule.
    sigma_path = os.path.join(DEFAULT_CONFIG.artifact_dir, 'sigma_data.json')
    if os.path.exists(sigma_path):
        with open(sigma_path) as fh:
            config['sigma_data'] = float(json.load(fh)['occupied'])
    config['P_mean'] = math.log(config['sigma_data'])

    # Nest under the existing dirs so the GCS mounts and rsyncs pick it up unchanged.
    for key in ('checkpoint_dir', 'sample_dir'):
        head, tail = os.path.split(config[key])
        config[key] = os.path.join(head, DEFAULT_CONFIG.dataset, tail)
    config['data_dir'] = DEFAULT_CONFIG.artifact_dir
    return config


def get_config(mode='local'):
    """Return configuration dict for the specified training mode.

    Args:
        mode: 'local' (Mac, sanity checks) or 'diagnostic' (A100, actual remote runs)
    """
    if mode == 'local':
        config = CONFIG_LOCAL.copy()
    elif mode == 'diagnostic':
        config = CONFIG_DIAGNOSTIC.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'local' or 'diagnostic'.")

    if DEFAULT_CONFIG.dataset != GTEX_DATASET:
        config = _factorized_overlay(config)
    return config


def print_config(config):
    """Print configuration in a readable format."""
    print("\n" + "="*60)
    print("DIFFUSION MODEL CONFIGURATION")
    print("="*60)

    print("\n📐 Model Architecture:")
    print(f"  Image size: {config['image_size']}×{config['image_size']}")
    print(f"  Channels: {config['channels']}")
    print(f"  ResNet blocks per level: {config['num_res_blocks']}")
    print(f"  Attention at resolutions: {config['attention_resolutions']}")
    print(f"  Embedding dimension: {config['embedding_dim']}")
    if config.get('attributes'):
        print("  Conditioning: factorized")
        for name, size in config['attributes']:
            print(f"    {name:<10} {size:>3} codes + 1 null token")
    else:
        print(f"  Number of classes: {config['num_classes']}")
    print(f"  Residual balance: {config.get('res_balance', 0.3)} (skip/residual split)")

    print("\n🎯 Training:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}  schedule: {config.get('lr_schedule', 'cosine')}")
    print(f"  Total steps: {config['num_steps']:,}")
    print(f"  Warmup steps: {config['warmup_steps']:,}")
    print(f"  Diag interval: {config.get('diag_interval', 'disabled')}")
    print(f"  Mixed precision: {config['mixed_precision']}")
    print(f"  EMA decay: {config['ema_decay']}")

    print("\n🔀 EDM2 Noise:")
    print(f"  sigma_data: {config['sigma_data']}  (std of normalized training data)")
    print(f"  P_mean: {config['P_mean']}  P_std: {config['P_std']}  → median σ ≈ {config['P_mean']:.2f}|exp = {2.718**config['P_mean']:.3f}")
    print(f"  sigma range: [{config['sigma_min']}, {config['sigma_max']}]  rho: {config.get('sigma_rho', 7)}")
    print(f"  Classifier-free dropout: {config['dropout_rate']*100:.0f}%")
    print(f"  Act clip magnitude: ±{config.get('act_clip_magnitude', 256.0)}")

    print("\n💾 Data:")
    print(f"  Data directory: {config['data_dir']}")
    print(f"  Checkpoint directory: {config['checkpoint_dir']}")
    print(f"  Sample directory: {config['sample_dir']}")

    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='View diffusion model configurations')
    parser.add_argument('--mode', type=str, choices=['local', 'diagnostic'], default='local',
                        help='Configuration mode to display')
    args = parser.parse_args()

    config = get_config(args.mode)
    print(f"\n{args.mode.upper()} Configuration:")
    print_config(config)
