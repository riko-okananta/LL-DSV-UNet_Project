"""
Core building blocks for LL-DSV-UNet.

Components:
    - RDBNet  : Residual Dense Block Network with asymmetric dilated convolutions
    - contextual_attention : Channel + Spatial attention for skip connections
    - decoder_block        : Full decoder stage (attention + RDBNet)
    - upsample / downsample: Utility spatial resamplers
"""

import tensorflow as tf
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Residual Dense Block Network (RDB-Net)
# ---------------------------------------------------------------------------

def RDBNet(input_image, filters: int, dilation_rate):
    """
    Residual Dense Block with asymmetric multi-scale dilated convolutions.

    Architecture:
        - 1x1 pointwise projection (bottleneck)
        - 7x7 → 3x3 → 3x1 → 1x3 dilated convolutions with dense connections
        - Dropout (0.25) + 1x1 fusion + residual addition

    Args:
        input_image : tf.Tensor   Input feature map.
        filters     : int         Number of output channels.
        dilation_rate           : Dilation rate for all conv layers.

    Returns:
        tf.Tensor: Enhanced feature map, same spatial size as input.
    """
    inp = layers.Conv2D(filters // 4, (1, 1), padding="same")(input_image)

    x1 = layers.Conv2D(filters // 4, (7, 7), dilation_rate=dilation_rate, padding="same")(input_image)
    x1 = layers.Concatenate()([x1, inp])

    x2 = layers.Conv2D(filters // 4, (3, 3), dilation_rate=dilation_rate, padding="same")(x1)
    x2 = layers.Concatenate()([x2, x1, inp])

    x3 = layers.Conv2D(filters // 4, (3, 1), dilation_rate=dilation_rate, padding="same")(x2)
    x3 = layers.Concatenate()([x3, x2, x1, inp])

    x4 = layers.Conv2D(filters // 4, (1, 3), dilation_rate=dilation_rate, padding="same")(x3)
    x4 = layers.Concatenate()([x4, x3, x2, x1, inp])

    x = layers.Dropout(rate=0.25)(x4)
    x = layers.Conv2D(filters, (1, 1), activation="relu", padding="same")(x)
    output = layers.Add()([x, input_image])

    return output


# ---------------------------------------------------------------------------
# Contextual Attention Module (CAM)
# ---------------------------------------------------------------------------

def contextual_attention(encode, decode, filters: int):
    """
    Contextual Attention Module for skip-connection fusion.

    Combines encoder features with upsampled decoder features using:
        1. Channel attention  via Global Avg + Max Pooling → FC gates
        2. Spatial attention  via multi-scale dilated convolutions (r=1,3,7)

    Args:
        encode  : tf.Tensor  Encoder skip-connection feature map.
        decode  : tf.Tensor  Upsampled decoder feature map.
        filters : int        Number of channels for the fused output.

    Returns:
        tf.Tensor: Attended fused feature map.
    """
    decode = layers.Resizing(encode.shape[1], encode.shape[2])(decode)
    fusion = layers.Concatenate()([encode, decode])
    fusion = layers.Conv2D(filters, (1, 1), padding="same")(fusion)

    # --- Channel attention ---
    g_avg = layers.GlobalAveragePooling2D()(fusion)
    g_max = layers.GlobalMaxPooling2D()(fusion)
    glbl  = layers.Concatenate()([g_avg, g_max])
    glbl  = layers.Dense(filters // 2, activation="relu")(glbl)
    glbl  = layers.Dense(filters, activation="sigmoid")(glbl)
    w_ch  = layers.Reshape((1, 1, filters))(glbl)
    x1    = layers.Multiply()([fusion, w_ch])

    # --- Spatial attention (multi-scale dilated) ---
    dc1 = layers.Conv2D(filters, (3, 3), padding="same")(fusion)
    dc2 = layers.Conv2D(filters, (3, 3), dilation_rate=3,  padding="same")(fusion)
    dc3 = layers.Conv2D(filters, (3, 3), dilation_rate=7,  padding="same")(fusion)
    dc  = layers.Concatenate()([dc1, dc2, dc3])
    w_sp = layers.Conv2D(filters, (1, 1), activation="sigmoid")(dc)
    x2   = layers.Multiply()([fusion, w_sp])

    return layers.Add()([fusion, x1, x2])


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------

def decoder_block(decode, skip, filters: int):
    """
    Single decoder stage: Contextual Attention → Conv → GELU → RDBNet.

    Args:
        decode  : tf.Tensor  Feature map from previous decoder stage.
        skip    : tf.Tensor  Encoder skip-connection at matching resolution.
        filters : int        Number of output channels.

    Returns:
        tf.Tensor: Decoded feature map.
    """
    d = contextual_attention(skip, decode, filters)
    d = layers.Conv2D(filters, (3, 3), padding="same")(d)
    d = layers.Activation("gelu")(d)
    d = RDBNet(d, filters, (1, 1))
    return d


# ---------------------------------------------------------------------------
# Spatial Resampling Utilities
# ---------------------------------------------------------------------------

def upsample(inputs, num_iter: int):
    """
    Progressive 2x nearest-neighbor upsampling.

    Args:
        inputs   : tf.Tensor  Input tensor.
        num_iter : int        Number of 2x upsampling steps.

    Returns:
        tf.Tensor: Upsampled tensor (spatial size × 2^num_iter).
    """
    for _ in range(num_iter):
        inputs = layers.UpSampling2D((2, 2))(inputs)
    return inputs


def downsample(inputs, num: int, channels: int):
    """
    Progressive 2x max-pool downsampling followed by 1x1 channel projection.

    Args:
        inputs   : tf.Tensor  Input tensor.
        num      : int        Number of 2x pooling steps.
        channels : int        Output channel count after projection.

    Returns:
        tf.Tensor: Downsampled and channel-projected tensor.
    """
    for _ in range(num):
        inputs = layers.MaxPooling2D((2, 2))(inputs)
    return layers.Conv2D(channels, (1, 1), padding="same")(inputs)
