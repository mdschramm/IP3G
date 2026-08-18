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
        # dtype='float32': GroupNorm computes (x-μ)/σ, which requires precise
        # subtraction. In FP16, activations >~1000 cause catastrophic cancellation
        # (FP16 has 3 decimal digits of precision — x-μ loses all signal when
        # x≈μ≈5000). Once GN stops normalizing, each block doubles rather than
        # normalizes, driving late-decoder activations exponential. FP32 here is
        # negligible cost (GN is <0.1% of total FLOPs).
        self.group_norm = layers.GroupNormalization(groups=self.num_groups, dtype='float32')
        # Zero-init: scale_raw=0 → scale=1 (passthrough) at init. Forces
        # conditioning to earn its influence via gradients rather than starting
        # with random scale. Prevents tanh saturation by keeping scale_raw near
        # zero unless loss explicitly requires deviation. (DiT adaLN-Zero pattern)
        self.scale_shift_mlp = layers.Dense(
            self.channels * 2,
            kernel_initializer='zeros',
        )
        super().build(input_shape)
        
    def call(self, inputs):
        x, conditioning = inputs
        
        # Normalize
        h = self.group_norm(x)  # [B, H, W, C]
        
        # Get scale and shift from conditioning
        scale_shift = self.scale_shift_mlp(conditioning)  # [B, C*2]
        scale_raw, shift = tf.split(scale_shift, 2, axis=-1)  # Each [B, C]

        # Cast to float32: group_norm output is float32 (dtype='float32'), but
        # scale_shift_mlp runs in the global compute dtype (float16 under mixed precision).
        # Mismatched dtypes cause a Mul op failure; casting keeps modulation arithmetic
        # in full precision, matching the GN output.
        scale = tf.cast(scale_raw, tf.float32)
        shift = tf.cast(shift, tf.float32)

        # Reshape for broadcasting: [B, C] → [B, 1, 1, C]
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]

        # adaLN-Zero: (1 + scale)*h + shift so scale_raw=0 at init is identity passthrough,
        # not zero-suppression. Cast back to input dtype (FP16 under mixed precision) so
        # downstream MPConv2D receives FP16 and computes in FP16 as intended.
        return tf.cast((1.0 + scale) * h + shift, x.dtype)
        
    
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
    
    def __init__(self, channels, dropout=0.1, res_balance=0.3, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.dropout_rate = dropout
        self.res_balance = float(res_balance)
        self._mp_denom = float(np.sqrt((1.0 - res_balance) ** 2 + res_balance ** 2))
        
    def build(self, input_shape):
        # input_shape is a list: [x_shape, conditioning_shape]
        x_shape = input_shape[0]
        in_channels = x_shape[-1]

        # First conv path — pre-norm on INPUT channels, then project to output channels.
        self.adagn1 = AdaGN(in_channels)
        self.act1 = MPSiLU()
        self.conv1 = MPConv2D(self.channels, 3)

        # Second conv path
        self.adagn2 = AdaGN(self.channels)
        self.act2 = MPSiLU()
        self.dropout = layers.Dropout(self.dropout_rate)
        self.conv2 = MPConv2D(self.channels, 3)

        # Skip connection (if input channels != output channels)
        if x_shape[-1] != self.channels:
            self.skip_conv = MPConv2D(self.channels, 1)
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

        # Asymmetric magnitude-preserving sum: (1-t)*skip + t*residual, normalised by
        # sqrt((1-t)²+t²) so output variance equals input variance. t=0.3 (70/30) matches
        # EDM2 res_balance default and is more conservative than the symmetric t=0.5 case.
        return ((1.0 - self.res_balance) * x + self.res_balance * h) / self._mp_denom
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'dropout': self.dropout_rate,
            'res_balance': self.res_balance,
        })
        return config


class QKNormMultiHeadAttention(layers.MultiHeadAttention):
    """MultiHeadAttention with LayerNorm applied to Q and K after projection.

    Problem: Q and K projection weights (W_q, W_k) grow in magnitude during
    training alongside activations. Attention logits scale as M² (Q·Kᵀ / √d),
    so even modest weight growth produces logits of ~98+, collapsing softmax to
    a hard argmax and zeroing out gradients through attention (observed at step
    1600 in diagnostic run: mean activation 4.2 → 23.6, loss 0.09 → 0.248).

    Fix: after Keras projects h → Q and h → K (shape [B, seq_len, heads, head_dim]),
    normalize each head vector to zero mean, unit variance via LayerNorm(axis=-1).
    This caps logit magnitude at ~√head_dim regardless of W_q / W_k scale.

    Compute overhead: ~0.6% of attention block FLOPs (LayerNorm is O(seq_len × C),
    attention scores are O(seq_len² × C)).

    Reference: Zhai et al. 2022, "Scaling Vision Transformers to 22 Billion
    Parameters" — found QK-norm necessary (not optional) for stable training at scale.
    """

    def build(self, query_shape, value_shape, key_shape=None):
        super().build(query_shape, value_shape, key_shape)
        # Normalize over the per-head key dimension (axis=-1 of the projected
        # [B, seq_len, num_heads, head_dim] tensors) — one norm per head independently.
        # dtype='float32': same FP16 catastrophic cancellation risk as GroupNorm.
        self._q_layer_norm = layers.LayerNormalization(axis=-1, dtype='float32')
        self._k_layer_norm = layers.LayerNormalization(axis=-1, dtype='float32')

    def _compute_attention(self, query, key, value, *args, **kwargs):
        # query, key shape: [B, seq_len, num_heads, head_dim] (post-projection, pre-scores)
        # Normalize each head's Q and K to unit variance before computing Q·Kᵀ / √d.
        # Without this, logits scale as ‖W_q‖·‖W_k‖ (grows quadratically with weight
        # magnitude), pushing softmax into a hard argmax that freezes attention gradients.
        # *args/**kwargs: forward-compatible with Keras 3 which passes extra positional args.
        #
        # LayerNorm is dtype='float32' so its output is float32. Cast back to the
        # incoming dtype (float16 under mixed precision) so query/key/value are all
        # consistent when super()._compute_attention runs attn_weights @ value.
        orig_dtype = query.dtype
        query = tf.cast(self._q_layer_norm(query), orig_dtype)
        key   = tf.cast(self._k_layer_norm(key),   orig_dtype)
        return super()._compute_attention(query, key, value, *args, **kwargs)


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
        self.group_norm = layers.GroupNormalization(groups=min(32, channels), dtype='float32')
        # QKNormMultiHeadAttention applies LayerNorm to Q and K after projection
        # to prevent attention logit explosion as weight magnitudes grow.
        self.attention = QKNormMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=channels // self.num_heads
        )
        super().build(input_shape)

    def call(self, x):
        batch, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Normalize in float32, then cast back to compute dtype before attention.
        # group_norm is dtype='float32'; explicitly casting here (rather than relying on
        # QKNormMultiHeadAttention.__call__ to auto-cast) keeps this consistent with
        # SparseSelfAttention and avoids fragile implicit behavior in the MHA subclass.
        h = tf.cast(tf.reshape(self.group_norm(x), [batch, height * width, channels]), x.dtype)

        # Self-attention
        h = self.attention(h, h)
        
        # Reshape back: [B, H*W, C] → [B, H, W, C]
        h = tf.reshape(h, [batch, height, width, channels])

        # Magnitude-preserving residual: (x + h) / sqrt(2).
        # Plain x + h amplifies by sqrt(2) when x and h have equal magnitude (as at init),
        # and grows with x when h has smaller magnitude (attention of GroupNorm output ≈ 1).
        # Division by sqrt(2) preserves magnitude for uncorrelated equal-magnitude inputs,
        # and attenuates when x >> h (stabilizing growth across layers).
        return (x + h) * tf.cast(2.0 ** -0.5, x.dtype)

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
        self.group_norm = layers.GroupNormalization(groups=min(32, channels), dtype='float32')
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

        # Normalize in float32, then cast back to compute dtype before gating/attention.
        # group_norm is dtype='float32' so h is float32; keep_f is cast to x.dtype (float16
        # under mixed precision). Without this cast, seq * keep_f is float32 × float16 → error.
        h = self.group_norm(x)
        seq = tf.cast(tf.reshape(h, [batch, height * width, channels]), x.dtype)
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
        return (x + attended) * tf.cast(2.0 ** -0.5, x.dtype)  # magnitude-preserving residual

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
        
        # Time embedding MLP — MPLinear preserves magnitude through the projection
        self.time_mlp = keras.Sequential([
            MPLinear(self.embedding_dim * 4),
            MPSiLU(),
            MPLinear(self.embedding_dim),
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
        self.conv = MPConv2D(self.channels, 3, strides=2)
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
        self.conv = MPConv2D(self.channels, 3)
        super().build(input_shape)
        
    def call(self, x):
        return self.conv(self.upsample(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config


class MPConv2D(layers.Layer):
    """Magnitude-preserving Conv2D (EDM2 Config D, §B.4).

    Normalizes each output filter to unit L2 norm in the forward pass.
    Formula: ŵ_i = w_i / (‖w_i‖₂ + ε), ε = 1e-4 (as in EDM2 paper).
    This prevents conv weight growth from amplifying activations across training,
    which is the root cause of the decoder activation explosion seen in diagnostic runs.

    The underlying parameter `kernel` is trained normally via gradient descent; the
    normalization is applied on-the-fly in call() so it never needs to be undone.
    """

    def __init__(self, filters, kernel_size, strides=1, padding='same', use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias

    def build(self, input_shape):
        kh, kw = self.kernel_size
        c_in = int(input_shape[-1])
        self.kernel = self.add_weight(
            shape=(kh, kw, c_in, self.filters), initializer='glorot_uniform', name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,), initializer='zeros', name='bias')
        super().build(input_shape)

    def call(self, x):
        w = tf.cast(self.kernel, tf.float32)
        # Flatten to [fan_in, filters], normalize each column (output filter) to unit norm.
        w_flat = tf.reshape(w, [-1, self.filters])
        w_norm = w_flat / (tf.norm(w_flat, axis=0, keepdims=True) + 1e-4)
        w_norm = tf.reshape(w_norm, self.kernel.shape)
        w_norm = tf.cast(w_norm, x.dtype)
        y = tf.nn.conv2d(x, w_norm,
                         strides=[1, self.strides, self.strides, 1],
                         padding=self.padding)
        if self.use_bias:
            y = y + tf.cast(self.bias, x.dtype)
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'kernel_size': self.kernel_size,
                    'strides': self.strides, 'padding': self.padding.lower(),
                    'use_bias': self.use_bias})
        return cfg


class MPLinear(layers.Layer):
    """Magnitude-preserving Dense layer (EDM2 Config D, §B.4).

    Same principle as MPConv2D: normalizes each output neuron's weight vector to
    unit norm. Formula: ŵ_i = w_i / (‖w_i‖₂ + ε), ε = 1e-4 (as in EDM2 paper).
    """

    def __init__(self, units, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        fan_in = int(input_shape[-1])
        self.kernel = self.add_weight(
            shape=(fan_in, self.units), initializer='glorot_uniform', name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,), initializer='zeros', name='bias')
        super().build(input_shape)

    def call(self, x):
        w = tf.cast(self.kernel, tf.float32)
        w_norm = w / (tf.norm(w, axis=0, keepdims=True) + 1e-4)
        w_norm = tf.cast(w_norm, x.dtype)
        y = tf.matmul(x, w_norm)
        if self.use_bias:
            y = y + tf.cast(self.bias, x.dtype)
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units, 'use_bias': self.use_bias})
        return cfg


class MPSiLU(layers.Layer):
    """Magnitude-preserving SiLU activation (EDM2 §B).

    Divides SiLU(x) = x·σ(x) by 0.596, the expected RMS of silu(x) for
    x ~ N(0,1), so variance is preserved through the activation. Completes
    the MP chain alongside MPConv2D, MPLinear, and the (x+h)/√2 residual.

    Mixed precision: casts the scalar divisor to x.dtype so output dtype
    matches input (float16 under mixed precision, float32 otherwise).
    """

    def call(self, x):
        return tf.keras.activations.swish(x) / tf.cast(0.596, x.dtype)

    def get_config(self):
        return super().get_config()


class MPFourier(layers.Layer):
    """Fixed random Fourier features for a scalar input (EDM2 §B.3).

    Maps a scalar c_noise = ln(σ)/4 to a fixed random feature vector:
        y = cos(x · freqs + phases) · sqrt(2)

    Frequencies and phases are sampled once at build time and are NOT trained.
    The sqrt(2) factor makes E[y²]=1 for any input distribution, so the feature
    vector is magnitude-preserving regardless of the input range.

    Used by the logvar branch to give it a rich, noise-level-dependent feature
    without coupling it to the spatial decoder activations.
    """

    def __init__(self, num_channels, bandwidth=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.bandwidth = float(bandwidth)

    def build(self, input_shape):
        freqs_init = (2 * np.pi * np.random.randn(self.num_channels) * self.bandwidth).astype(np.float32)
        phases_init = (2 * np.pi * np.random.uniform(size=self.num_channels)).astype(np.float32)
        self.freqs  = self.add_weight(shape=(self.num_channels,),
                                      initializer=tf.keras.initializers.Constant(freqs_init),
                                      trainable=False, name='freqs',  dtype=tf.float32)
        self.phases = self.add_weight(shape=(self.num_channels,),
                                      initializer=tf.keras.initializers.Constant(phases_init),
                                      trainable=False, name='phases', dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        # Explicitly cast to float32: Keras mixed-precision policy auto-casts non-trainable
        # weights to the global compute dtype (float16) inside call(). Without explicit casts
        # x_f32 (float32) * self.freqs (float16) produces a dtype-mismatch error.
        x_f32     = tf.cast(x,           tf.float32)  # [B]
        freqs_f32 = tf.cast(self.freqs,  tf.float32)  # [num_channels]
        phases_f32= tf.cast(self.phases, tf.float32)  # [num_channels]
        y = x_f32[:, None] * freqs_f32[None, :]       # [B, num_channels]
        y = y + phases_f32[None, :]
        y = tf.cos(y) * tf.cast(np.sqrt(2), tf.float32)
        return y                                       # [B, num_channels], float32

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_channels': self.num_channels, 'bandwidth': self.bandwidth})
        return cfg


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
    act_clip = float(config.get('act_clip_magnitude', 256.0))
    num_classes = config['num_classes']
    use_sparse_attention = config.get('use_sparse_attention', False)
    sparse_top_k_frac = config.get('sparse_top_k_frac', 0.5)
    
    res_balance = config.get('res_balance', 0.3)

    # Inputs — t_input is float32: receives c_noise = ln(σ)/4 (continuous EDM2 conditioning)
    x_input    = layers.Input(shape=(image_size, image_size, in_channels), name='x_noisy')
    t_input    = layers.Input(shape=(), dtype=tf.float32, name='timesteps')
    c_input    = layers.Input(shape=(), dtype=tf.int32, name='class_labels')
    mask_input = layers.Input(shape=(image_size, image_size, in_channels),
                              dtype=tf.float32, name='occupancy_mask')

    # Embeddings
    embedding_layer = TimeAndClassEmbedding(num_classes, embedding_dim)
    conditioning = embedding_layer([t_input, c_input])

    # Initial convolution + soft mask conditioning
    h = MPConv2D(channels[0], 3)(x_input)
    # Project binary mask into feature space (use_bias=False so unoccupied positions
    # contribute exactly zero, not a learned offset).
    h = h + MPConv2D(channels[0], 1, use_bias=False)(tf.cast(mask_input, h.dtype))
    
    # Encoder
    skip_connections = []
    current_resolution = image_size
    
    for level, ch in enumerate(channels):
        # ResNet blocks
        for _ in range(num_res_blocks):
            h = ResNetBlock(ch, dropout, res_balance=res_balance)([h, conditioning])
        
        # Self-attention at specified resolutions
        if current_resolution in attention_resolutions:
            h = _make_attention(num_heads, use_sparse_attention, sparse_top_k_frac)(h)
        
        # EDM2 §B: clamp activations to ±act_clip at the end of each encoder block.
        # Prevents rare FP16 overflow spikes from propagating through skip connections.
        h = keras.ops.clip(h, -act_clip, act_clip)
        skip_connections.append(h)

        # Downsample (except at last level)
        if level < len(channels) - 1:
            h = Downsample(channels[level + 1])(h)
            current_resolution //= 2
    
    # Bottleneck
    h = ResNetBlock(channels[-1], dropout, res_balance=res_balance)([h, conditioning])
    h = _make_attention(num_heads, use_sparse_attention, sparse_top_k_frac)(h)
    h = ResNetBlock(channels[-1], dropout, res_balance=res_balance)([h, conditioning])

    # Logvar head — EDM2 §B.3 / §5: a separate MLP that takes only c_noise (t_input),
    # NOT the spatial bottleneck features. This is critical: logvar should depend on the
    # noise level σ alone, not on spatial content. Branching from bottleneck features
    # entangles logvar with the main prediction, causing Adam to overshoot logvar negative
    # near peak LR and producing mse/exp(logvar) → 1e8.
    #
    # Architecture: MPFourier(c_noise) → fixed random Fourier features → MPLinear → scalar
    # MPLinear keeps weight norm = 1, so logvar output is bounded by ||fourier_features||₂.
    logvar_channels = config.get('logvar_channels', 128)
    logvar = MPFourier(logvar_channels)(t_input)          # [B, logvar_channels], float32
    logvar = MPLinear(1)(logvar)                          # [B, 1], float32
    logvar = layers.Reshape((1, 1, 1))(logvar)            # [B, 1, 1, 1] — broadcasts over H×W×C

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
            h = ResNetBlock(ch, dropout, res_balance=res_balance)([h, conditioning])
        
        # Self-attention at specified resolutions
        if current_resolution in attention_resolutions:
            h = _make_attention(num_heads, use_sparse_attention, sparse_top_k_frac)(h)

        # EDM2 §B: clamp activations to ±act_clip at the end of each decoder block.
        h = keras.ops.clip(h, -act_clip, act_clip)

    # Output projection
    out_channels = h.shape[-1]
    h = layers.GroupNormalization(groups=min(32, out_channels), dtype='float32')(h)
    h = MPSiLU()(h)
    # MPConv2D output dtype matches h, which is float32 from the preceding GroupNorm.
    output = MPConv2D(in_channels, 3)(h)
    # Hard occupancy enforcement: zero out positions that are structurally empty.
    # Paired with the masked loss in train_step, this guarantees the model never
    # predicts anything at unoccupied channels and is never penalized there.
    output = output * tf.cast(mask_input, output.dtype)

    # Build model — outputs [pred_F, logvar]; logvar is [B,1,1,1], pred_F is [B,H,W,C]
    model = keras.Model(
        inputs=[x_input, t_input, c_input, mask_input],
        outputs=[output, logvar],
        name='conditional_unet'
    )
    
    return model


if __name__ == "__main__":
    # Test model building
    from diffusion.diffusion_config import get_config
    
    print("Testing U-Net model construction...")
    
    # Test with specific config
    config = get_config('diagnostic')
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
    in_ch = config['in_channels']
    x_test    = tf.random.normal([batch_size, 128, 128, in_ch])
    t_test    = tf.random.uniform([batch_size], -1.55, 1.10, dtype=tf.float32)  # c_noise = ln(sigma)/4
    c_test    = tf.random.uniform([batch_size], 0, 54, dtype=tf.int32)
    mask_test = tf.ones([batch_size, 128, 128, in_ch], dtype=tf.float32)

    print(f"Input shapes:")
    print(f"  x_noisy:        {x_test.shape}")
    print(f"  timesteps:      {t_test.shape}")
    print(f"  class_labels:   {c_test.shape}")
    print(f"  occupancy_mask: {mask_test.shape}")

    pred_F, logvar = model([x_test, t_test, c_test, mask_test], training=False)
    print(f"\nOutput shape: {pred_F.shape}  logvar shape: {logvar.shape}")
    print(f"Output range: [{tf.reduce_min(pred_F):.4f}, {tf.reduce_max(pred_F):.4f}]")
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    print("\n✅ Model construction and forward pass successful!")
