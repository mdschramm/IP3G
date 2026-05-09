"""
Conditional U-Net with Adaptive Group Normalization for DDPM.

Implements:
- AdaGN (Adaptive Group Normalization) for conditioning
- Time and class embeddings
- ResNet blocks with skip connections
- Self-attention layers
- U-Net encoder-decoder architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_sinusoidal_embeddings(timesteps, embedding_dim):
    """
    Sinusoidal positional embeddings for timesteps.
    
    Args:
        timesteps: Tensor of shape [batch_size]
        embedding_dim: Dimension of embeddings
        
    Returns:
        Embeddings of shape [batch_size, embedding_dim]
    """
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.cast(timesteps, tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb


class AdaGN(layers.Layer):
    """
    Adaptive Group Normalization.
    
    Modulates normalized features using conditioning signal (time + class).
    """
    
    def __init__(self, channels, num_groups=32, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_groups = min(num_groups, channels)  # Ensure groups <= channels
        
    def build(self, input_shape):
        # input_shape is a list: [x_shape, conditioning_shape]
        self.group_norm = layers.GroupNormalization(groups=self.num_groups)
        self.scale_shift_mlp = layers.Dense(self.channels * 2)
        super().build(input_shape)
        
    def call(self, inputs):
        x, conditioning = inputs
        
        # Normalize
        h = self.group_norm(x)  # [B, H, W, C]
        
        # Get scale and shift from conditioning
        scale_shift = self.scale_shift_mlp(conditioning)  # [B, C*2]
        scale, shift = tf.split(scale_shift, 2, axis=-1)  # Each [B, C]
        
        # Reshape for broadcasting: [B, C] → [B, 1, 1, C]
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        
        # Modulate: scale * normalized + shift
        return scale * h + shift
        
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_groups': self.num_groups,
        })
        return config


class ResNetBlock(layers.Layer):
    """
    ResNet block with Adaptive Group Normalization.
    
    Two conv layers with AdaGN modulation and skip connection.
    """
    
    def __init__(self, channels, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        # input_shape is a list: [x_shape, conditioning_shape]
        x_shape = input_shape[0]
        in_channels = x_shape[-1]

        # First conv path — pre-norm on INPUT channels, then project to output channels.
        self.adagn1 = AdaGN(in_channels)
        self.act1 = layers.Activation('swish')
        self.conv1 = layers.Conv2D(self.channels, 3, padding='same')
        
        # Second conv path
        self.adagn2 = AdaGN(self.channels)
        self.act2 = layers.Activation('swish')
        self.dropout = layers.Dropout(self.dropout_rate)
        self.conv2 = layers.Conv2D(self.channels, 3, padding='same')
        
        # Skip connection (if input channels != output channels)
        if x_shape[-1] != self.channels:
            self.skip_conv = layers.Conv2D(self.channels, 1)
        else:
            self.skip_conv = None
            
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x, conditioning = inputs
        
        # First conv
        h = self.adagn1([x, conditioning])
        h = self.act1(h)
        h = self.conv1(h)
        
        # Second conv
        h = self.adagn2([h, conditioning])
        h = self.act2(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)
        
        # Skip connection
        if self.skip_conv is not None:
            x = self.skip_conv(x)
            
        return x + h
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'dropout': self.dropout_rate,
        })
        return config


class SelfAttention(layers.Layer):
    """
    Multi-head self-attention layer.
    
    Allows spatial locations to attend to each other.
    """
    
    def __init__(self, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.group_norm = layers.GroupNormalization(groups=min(32, channels))
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=channels // self.num_heads
        )
        super().build(input_shape)
        
    def call(self, x):
        batch, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Normalize
        h = self.group_norm(x)
        
        # Reshape to sequence: [B, H, W, C] → [B, H*W, C]
        h = tf.reshape(h, [batch, height * width, channels])
        
        # Self-attention
        h = self.attention(h, h)
        
        # Reshape back: [B, H*W, C] → [B, H, W, C]
        h = tf.reshape(h, [batch, height, width, channels])
        
        return x + h  # Residual connection
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads})
        return config


class SparseSelfAttention(layers.Layer):
    """Top-k masked multi-head self-attention.

    For sparse data (e.g., 55% near-zero pixels), dense attention is dominated by
    background tokens. This layer ranks tokens by L2 magnitude on the (normalized)
    feature map, keeps only the top `top_k_frac` fraction, and masks the rest with
    -inf in the attention logits so they neither contribute keys nor receive queries.

    Mask is computed in the forward pass (no learned mask predictor) — gradients
    still flow through the surviving tokens via the residual connection.
    """

    def __init__(self, num_heads=4, top_k_frac=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.top_k_frac = float(top_k_frac)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.group_norm = layers.GroupNormalization(groups=min(32, channels))
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=channels // self.num_heads
        )
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]

        # Normalize and flatten to sequence: [B, H, W, C] -> [B, H*W, C]
        h = self.group_norm(x)
        seq = tf.reshape(h, [batch, height * width, channels])
        seq_len = height * width

        # Per-token magnitude (L2 over channels); shape [B, H*W]
        # Compute in float32 for numerical stability even when input is float16.
        magnitudes = tf.norm(tf.cast(seq, tf.float32), axis=-1)

        # Number of tokens to keep (at least 1)
        k = tf.cast(
            tf.maximum(tf.cast(seq_len, tf.float32) * self.top_k_frac, 1.0),
            tf.int32,
        )

        # Threshold = k-th largest magnitude per batch element; tokens >= threshold survive.
        # Cast keep_f to x.dtype (float16 under mixed precision, float32 otherwise) so
        # the multiplicative gating below doesn't produce a dtype mismatch.
        topk_vals, _ = tf.math.top_k(magnitudes, k=k)            # [B, k]
        threshold = topk_vals[:, -1:]                            # [B, 1]
        keep_f = tf.cast(magnitudes >= threshold, x.dtype)       # [B, H*W] in {0., 1.}
        keep_f = keep_f[:, :, None]                              # [B, T, 1] for broadcasting

        # Multiplicative gating instead of an attention_mask. Semantics:
        #   - Zeroing Q/K/V of non-kept tokens makes their V contribution to other
        #     tokens' weighted sums exactly 0 (regardless of softmax weight).
        #   - Zeroing the output for non-kept query positions ensures they emit no
        #     residual update.
        # Net effect: only top-k tokens influence and are influenced by attention,
        # matching the intent of a hard top-k mask without any bool broadcasting.
        seq_gated = seq * keep_f
        attended = self.attention(seq_gated, seq_gated)
        attended = attended * keep_f

        # Reshape back: [B, H*W, C] -> [B, H, W, C]
        attended = tf.reshape(attended, [batch, height, width, channels])
        return x + attended  # residual connection (same as dense version)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'top_k_frac': self.top_k_frac,
        })
        return config


def _make_attention(num_heads, use_sparse, top_k_frac):
    """Factory: return a SparseSelfAttention if use_sparse else SelfAttention."""
    if use_sparse:
        return SparseSelfAttention(num_heads=num_heads, top_k_frac=top_k_frac)
    return SelfAttention(num_heads=num_heads)


class TimeAndClassEmbedding(layers.Layer):
    """
    Combined time and class embeddings.
    
    Creates embeddings for timesteps and class labels, then combines them.
    """
    
    def __init__(self, num_classes, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
    def build(self, input_shape):
        # Class embedding layer (+1 for unconditional token)
        self.class_embedding = layers.Embedding(
            self.num_classes + 1,
            self.embedding_dim
        )
        
        # Time embedding MLP
        self.time_mlp = keras.Sequential([
            layers.Dense(self.embedding_dim * 4, activation='swish'),
            layers.Dense(self.embedding_dim)
        ])
        super().build(input_shape)
        
    def call(self, inputs):
        timesteps, class_labels = inputs
        
        # Time embedding
        time_emb = get_sinusoidal_embeddings(timesteps, self.embedding_dim)
        time_emb = self.time_mlp(time_emb)  # [B, emb_dim]
        
        # Class embedding
        class_emb = self.class_embedding(class_labels)  # [B, emb_dim]
        
        # Combine (element-wise addition)
        return time_emb + class_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
        })
        return config


class Downsample(layers.Layer):
    """Downsampling layer using strided convolution."""
    
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(self.channels, 3, strides=2, padding='same')
        super().build(input_shape)
        
    def call(self, x):
        return self.conv(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config


class Upsample(layers.Layer):
    """Upsampling layer using transposed convolution."""
    
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        
    def build(self, input_shape):
        # Nearest-neighbor upsample + conv to avoid checkerboard artifacts
        self.upsample = layers.UpSampling2D(size=2, interpolation='nearest')
        self.conv = layers.Conv2D(self.channels, 3, padding='same')
        super().build(input_shape)
        
    def call(self, x):
        return self.conv(self.upsample(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config


def build_unet(config):
    """
    Build conditional U-Net model for DDPM.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Keras Model that takes [x_noisy, timesteps, class_labels] and outputs predicted noise
    """
    # Extract config
    image_size = config['image_size']
    in_channels = config['in_channels']
    channels = config['channels']
    num_res_blocks = config['num_res_blocks']
    attention_resolutions = config['attention_resolutions']
    num_heads = config['num_heads']
    dropout = config['dropout']
    embedding_dim = config['embedding_dim']
    num_classes = config['num_classes']
    use_sparse_attention = config.get('use_sparse_attention', False)
    sparse_top_k_frac = config.get('sparse_top_k_frac', 0.5)
    
    # Inputs
    x_input = layers.Input(shape=(image_size, image_size, in_channels), name='x_noisy')
    t_input = layers.Input(shape=(), dtype=tf.int32, name='timesteps')
    c_input = layers.Input(shape=(), dtype=tf.int32, name='class_labels')
    
    # Embeddings
    embedding_layer = TimeAndClassEmbedding(num_classes, embedding_dim)
    conditioning = embedding_layer([t_input, c_input])
    
    # Initial convolution
    h = layers.Conv2D(channels[0], 3, padding='same')(x_input)
    
    # Encoder
    skip_connections = []
    current_resolution = image_size
    
    for level, ch in enumerate(channels):
        # ResNet blocks
        for _ in range(num_res_blocks):
            h = ResNetBlock(ch, dropout)([h, conditioning])
        
        # Self-attention at specified resolutions
        if current_resolution in attention_resolutions:
            h = _make_attention(num_heads, use_sparse_attention, sparse_top_k_frac)(h)
        
        # Save skip connection
        skip_connections.append(h)
        
        # Downsample (except at last level)
        if level < len(channels) - 1:
            h = Downsample(channels[level + 1])(h)
            current_resolution //= 2
    
    # Bottleneck
    h = ResNetBlock(channels[-1], dropout)([h, conditioning])
    h = _make_attention(num_heads, use_sparse_attention, sparse_top_k_frac)(h)
    h = ResNetBlock(channels[-1], dropout)([h, conditioning])
    
    # Decoder
    for level in reversed(range(len(channels))):
        ch = channels[level]
        
        # Upsample (except at first level in decoder)
        if level < len(channels) - 1:
            h = Upsample(ch)(h)
            current_resolution *= 2
        
        # Concatenate skip connection
        skip = skip_connections.pop()
        h = layers.Concatenate()([h, skip])
        
        # ResNet blocks
        for _ in range(num_res_blocks):
            h = ResNetBlock(ch, dropout)([h, conditioning])
        
        # Self-attention at specified resolutions
        if current_resolution in attention_resolutions:
            h = _make_attention(num_heads, use_sparse_attention, sparse_top_k_frac)(h)
    
    # Output projection
    out_channels = h.shape[-1]
    h = layers.GroupNormalization(groups=min(32, out_channels))(h)
    h = layers.Activation('swish')(h)
    # Force final output to float32 so the loss is always computed in fp32
    # under mixed precision training (no-op when policy is float32).
    output = layers.Conv2D(in_channels, 3, padding='same', dtype='float32')(h)
    
    # Build model
    model = keras.Model(
        inputs=[x_input, t_input, c_input],
        outputs=output,
        name='conditional_unet'
    )
    
    return model


if __name__ == "__main__":
    # Test model building
    from diffusion.diffusion_config import get_config
    
    print("Testing U-Net model construction...")
    
    # Test with local config
    config = get_config('remote')
    model = build_unet(config)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    # Test forward pass
    print("\n" + "="*60)
    print("TESTING FORWARD PASS")
    print("="*60)
    
    batch_size = 4
    x_test = tf.random.normal([batch_size, 128, 128, 1])
    t_test = tf.random.uniform([batch_size], 0, 1000, dtype=tf.int32)
    c_test = tf.random.uniform([batch_size], 0, 54, dtype=tf.int32)
    
    print(f"Input shapes:")
    print(f"  x_noisy: {x_test.shape}")
    print(f"  timesteps: {t_test.shape}")
    print(f"  class_labels: {c_test.shape}")
    
    output = model([x_test, t_test, c_test], training=False)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{tf.reduce_min(output):.4f}, {tf.reduce_max(output):.4f}]")
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    print("\n✅ Model construction and forward pass successful!")
