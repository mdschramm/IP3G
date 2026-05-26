"""
Configuration for Conditional DDPM with Classifier-Free Guidance.

Provides separate configurations for local (Mac M2) and remote (A100) training.
"""

# Local configuration - Mac M2, 16GB RAM (architectural sanity check, ~2-3 hours)
CONFIG_LOCAL = {
    # Model architecture
    'image_size': 128,
    'in_channels': 1,
    'channels': [32, 64, 128],           # 3 levels (kept small for M2)
    'num_res_blocks': 2,                  # Per resolution
    'attention_resolutions': [],        # Bottleneck attention enabled to exercise sparse attn path
    'num_heads': 4,                       # For attention layers
    'dropout': 0.1,
    'embedding_dim': 256,                 # Time + class embeddings
    'num_classes': 54,                    # Gene expression classes
    'excluded_classes': [6, 24, 25, 31],  # Classes with <10 samples — too few for CFG conditioning
    'use_sparse_attention': False,         # Top-k masked attention (sparsity fix)
    'sparse_top_k_frac': 0.5,             # Attend to top 50% of tokens by magnitude
    
    # Training (~2-3 hours on M2)
    'batch_size': 8,                      # M1/M2 memory headroom
    'learning_rate': 1e-4,
    'lr_schedule': 'flat',                # warmup + constant; plateau = model, not LR→0
    'num_steps': 700,                   # Enough to validate loss trajectory + early structure
    'save_interval': 999999,
    'sample_interval': 250,               # Frequent so we can watch structure emerge
    'log_interval': 1,
    'diag_interval': 20,                  # Weight/activation magnitude logging interval
    'ema_decay': 0.99,                  # Lower so EMA tracks fast in short runs
    'gradient_clip': 1.0,
    'warmup_steps': 50,                  # Proportional to num_steps
    'mixed_precision': False,             # M2 doesn't benefit much

    # Diffusion
    'timesteps': 1000,
    'variance_schedule': 'cosine',
    'noise_timestep_range': [200, 400],      # t ∈ [t_min, t_max] for training and DDIM [t_start, t_end]

    # Preprocessing / sampling fixes
    'eps_threshold': 0.05,                # Soft-threshold predicted noise during sampling

    # Classifier-free guidance
    'dropout_rate': 0.10,                 # 15% unconditional dropout
    
    # Data
    'data_dir': 'output/preprocessing',
    'feature_file': 'resized_expressions.npy',
    'label_file': 'y_primary_disease_or_tissue.npy',
    'checkpoint_dir': 'output/diffusion/local/checkpoints',
    'sample_dir': 'output/diffusion/local/samples',
}

# Remote configuration - NVIDIA A100 40GB (Full Training)
CONFIG_REMOTE = {
    # Model architecture
    'image_size': 128,
    'in_channels': 1,
    'channels': [32, 64, 128, 256],      # 4 levels: 128→64→32→16 (bottleneck at 16×16)
    'num_res_blocks': 3,                  # 2 blocks per level (~6-8M params, ~750-1000 params/image)
    'attention_resolutions': [32],         # 32×32 in encoder+decoder; bottleneck always has 16×16 regardless
    'num_heads': 4,
    'dropout': 0.1,
    'embedding_dim': 256,                 # Scaled down with channels
    'num_classes': 54, # GTEX
    'excluded_classes': [6, 24, 25, 31],  # Classes with <10 samples — too few for CFG conditioning
    'use_sparse_attention': False,         # Top-k masked attention
    'sparse_top_k_frac': 0.5,

    # Training
    'batch_size': 128,                    # Large batch to reduce gradient variance at low-noise timesteps
    'learning_rate': 1e-4,
    'lr_schedule': 'cosine',
    'num_steps': 500_000,                  # Longer to let new architecture converge
    'save_interval': 10_000,
    'sample_interval': 5_000,
    'log_interval': 250,
    'diag_interval': 999_999,             # Disabled for remote — too costly
    'ema_decay': 0.9999,
    'gradient_clip': 1.0,
    'warmup_steps': 25_000,
    'mixed_precision': True,              # Use FP16 for speed

    # Diffusion
    'timesteps': 1000,
    'variance_schedule': 'cosine',
    'noise_timestep_range': [1, 200],      # t ∈ [t_min, t_max] for training and DDIM [t_start, t_end]

    # Preprocessing / sampling fixes
    'eps_threshold': 0.05,                # Soft-threshold predicted noise during sampling

    # Classifier-free guidance
    'dropout_rate': 0.10,

    # Data
    'data_dir': 'output/preprocessing',
    'feature_file': 'resized_expressions.npy',
    'label_file': 'y_primary_disease_or_tissue.npy',
    'checkpoint_dir': 'output/diffusion/remote/checkpoints',
    'sample_dir': 'output/diffusion/remote/samples',
}


# Diagnostic configuration — full remote architecture, 8000 steps
# Intent: run remotely for ~10 min to validate model behaviour (loss shape,
# CFG separation, gradient norms) before committing to a full 75k-step run.
CONFIG_DIAGNOSTIC = {
    # Architecture — identical to remote so results are representative
    'image_size': 128,
    'in_channels': 1,
    'channels': [32, 64, 128, 256],      # 4 levels: 128→64→32→16 (bottleneck at 16×16)
    'num_res_blocks': 3,                  # 2 blocks per level (~6-8M params, ~750-1000 params/image)
    'attention_resolutions': [32],         # 32×32 in encoder+decoder; bottleneck always has 16×16 regardless
    'num_heads': 4,
    'dropout': 0.1,
    'embedding_dim': 256,                 # Scaled down with channels
    'num_classes': 54, # GTEX
    'excluded_classes': [6, 24, 25, 31],  # Classes with <10 samples — too few for CFG conditioning
    'use_sparse_attention': False,         # Top-k masked attention
    'sparse_top_k_frac': 0.5,

    # Short training run — no intermediate checkpoints or sample images, only final model
    'batch_size': 32,
    'learning_rate': 1e-4,
    'lr_schedule': 'flat',                # warmup + constant; plateau = model, not LR→0
    'num_steps': 10_000,
    'save_interval': 999_999,         # Disabled: only the final save at end of train() fires
    'sample_interval': 2_000,
    'log_interval': 50,               # Very frequent — watch loss shape closely
    'diag_interval': 50,              # Weight/activation magnitude logging interval
    'ema_decay': 0.999,                # Fast EMA for short run
    'gradient_clip': 1.0,
    'warmup_steps': 500,              # 10% of num_steps (matches local proportion)
    'mixed_precision': True,          # Must match remote to catch FP16 issues early

    # Same diffusion settings as remote
    'timesteps': 1000,
    'variance_schedule': 'cosine',
    'noise_timestep_range': [1, 200],      # t ∈ [t_min, t_max] for training and DDIM [t_start, t_end]
    'eps_threshold': 0.05,

    # CFG
    'dropout_rate': 0.1,

    # Data
    'data_dir': 'output/preprocessing',
    'feature_file': 'resized_expressions.npy',
    'label_file': 'y_primary_disease_or_tissue.npy',
    'checkpoint_dir': 'output/diffusion/diagnostic/checkpoints',
    'sample_dir': 'output/diffusion/diagnostic/samples',
}


def get_config(mode='local'):
    """
    Get configuration for specified mode.
    
    Args:
        mode: 'local' for Mac M2 or 'remote' for A100
        
    Returns:
        Configuration dictionary
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
    
    print("\n🎯 Training:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}  schedule: {config.get('lr_schedule', 'cosine')}")
    print(f"  Total steps: {config['num_steps']:,}")
    print(f"  Warmup steps: {config['warmup_steps']:,}")
    print(f"  Diag interval: {config.get('diag_interval', 'disabled')}")
    print(f"  Mixed precision: {config['mixed_precision']}")
    print(f"  EMA decay: {config['ema_decay']}")
    
    print("\n🔀 Diffusion:")
    print(f"  Timesteps: {config['timesteps']}")
    print(f"  Schedule: {config['variance_schedule']}")
    ntr = config.get('noise_timestep_range', [1, config['timesteps']])
    print(f"  Noise timestep range: [{ntr[0]}, {ntr[1]}] (training t_min..t_max, DDIM t_end..t_start)")
    print(f"  Classifier-free dropout: {config['dropout_rate']*100:.0f}%")
    print(f"  Sparse attention: {config.get('use_sparse_attention', False)}"
          + (f" (top-{int(config.get('sparse_top_k_frac', 0.5)*100)}%)" if config.get('use_sparse_attention', False) else ""))
    print(f"  Sampling eps threshold: {config.get('eps_threshold', 0.0)}")
    
    print("\n💾 Data:")
    print(f"  Data directory: {config['data_dir']}")
    print(f"  Checkpoint directory: {config['checkpoint_dir']}")
    print(f"  Sample directory: {config['sample_dir']}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View diffusion model configurations')
    parser.add_argument('--mode', type=str, choices=['local', 'remote'], default='local',
                        help='Configuration mode to display')
    args = parser.parse_args()
    
    config = get_config(args.mode)
    print(f"\n{args.mode.upper()} Configuration:")
    print_config(config)
