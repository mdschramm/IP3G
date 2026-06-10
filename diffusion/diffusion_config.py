"""
Configuration for EDM2 Conditional Diffusion with Classifier-Free Guidance.

Provides separate configurations for local (Mac M2) and remote (A100) training.
Noise parameterization uses EDM2 (Karras et al. 2022/2023): continuous sigma
sampled from a log-normal, with c_skip/c_out/c_in preconditioning so the loss
weight w(σ)·c_out²=1 at all noise levels.
"""

# Local configuration - Mac M2, 16GB RAM (architectural sanity check, ~2-3 hours)
CONFIG_LOCAL = {
    # Model architecture
    'image_size': 128,
    'in_channels': 1,
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
    'sigma_data': 0.139,                  # std of normalized training data (measured: np.std(resized_expressions.npy))
    'P_mean': -2.0,                       # log-normal σ center: exp(-2.0)≈0.135≈sigma_data
    'P_std': 1.2,                         # log-normal σ spread (paper default)
    'sigma_min': 0.002,                   # near-clean inference endpoint
    'sigma_max': 80.0,                    # pure-noise inference starting point
    'sigma_rho': 7,                       # Karras step-density exponent (more steps near sigma_min)

    # Classifier-free guidance
    'dropout_rate': 0.10,

    # FP16 safety
    'act_clip_magnitude': 256.0,          # Clamp encoder/decoder block outputs to ±this value (EDM2 §B)

    # Data
    'data_dir': 'output/preprocessing',
    'feature_file': 'resized_expressions.npy',
    'label_file': 'y_primary_disease_or_tissue.npy',
    'checkpoint_dir': 'output/diffusion/local/checkpoints',
    'sample_dir': 'output/diffusion/local/samples',
}

# Remote configuration - NVIDIA GPU
CONFIG_REMOTE = {
    # Model architecture
    'image_size': 128,
    'in_channels': 1,
    'channels': [32, 64, 128, 256],      # 4 levels: 128→64→32→16 (bottleneck at 16×16)
    'num_res_blocks': 3,
    'attention_resolutions': [16],        # Attention at 16×16 bottleneck resolution
    'num_heads': 4,
    'dropout': 0.1,
    'embedding_dim': 256,
    'num_classes': 54,
    'excluded_classes': [6, 24, 25, 31],
    'use_sparse_attention': False,
    'sparse_top_k_frac': 0.5,
    'res_balance': 0.3,

    # Training
    'batch_size': 128,
    'learning_rate': 5e-5,
    'lr_schedule': 'cosine',
    'num_steps': 500_000,
    'save_interval': 10_000,
    'sample_interval': 5_000,
    'log_interval': 250,
    'diag_interval': 999_999,            # Disabled for remote — too costly
    'ema_decay': 0.9999,
    'gradient_clip': 1.0,
    'warmup_steps': 25_000,
    'mixed_precision': True,

    # EDM2 noise parameterization
    'sigma_data': 0.139,
    'P_mean': -2.0,
    'P_std': 1.2,
    'sigma_min': 0.002,
    'sigma_max': 80.0,
    'sigma_rho': 7,

    # Classifier-free guidance
    'dropout_rate': 0.10,

    # FP16 safety
    'act_clip_magnitude': 256.0,

    # Data
    'data_dir': 'output/preprocessing',
    'feature_file': 'resized_expressions.npy',
    'label_file': 'y_primary_disease_or_tissue.npy',
    'checkpoint_dir': 'output/diffusion/remote/checkpoints',
    'sample_dir': 'output/diffusion/remote/samples',
}


# Diagnostic configuration — full remote architecture, 30k steps
# Intent: run remotely to validate model behaviour (loss shape, CFG separation,
# gradient norms) before committing to a full 500k-step run.
CONFIG_DIAGNOSTIC = {
    # Architecture — identical to remote so results are representative
    'image_size': 128,
    'in_channels': 1,
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
    'sigma_data': 0.139,
    'P_mean': -2.0,
    'P_std': 1.2,
    'sigma_min': 0.002,
    'sigma_max': 80.0,
    'sigma_rho': 7,

    # CFG
    'dropout_rate': 0.1,

    # FP16 safety
    'act_clip_magnitude': 256.0,

    # Data
    'data_dir': 'output/preprocessing',
    'feature_file': 'resized_expressions.npy',
    'label_file': 'y_primary_disease_or_tissue.npy',
    'checkpoint_dir': 'output/diffusion/diagnostic/checkpoints',
    'sample_dir': 'output/diffusion/diagnostic/samples',
}


def get_config(mode='local'):
    """Return configuration dict for the specified training mode.

    Args:
        mode: 'local' (Mac M2), 'remote' (A100), or 'diagnostic' (30k-step remote arch)
    """
    if mode == 'local':
        return CONFIG_LOCAL.copy()
    elif mode == 'remote':
        return CONFIG_REMOTE.copy()
    elif mode == 'diagnostic':
        return CONFIG_DIAGNOSTIC.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'local', 'remote', or 'diagnostic'.")


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
    parser.add_argument('--mode', type=str, choices=['local', 'remote', 'diagnostic'], default='local',
                        help='Configuration mode to display')
    args = parser.parse_args()

    config = get_config(args.mode)
    print(f"\n{args.mode.upper()} Configuration:")
    print_config(config)
