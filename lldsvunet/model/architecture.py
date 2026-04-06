"""
LL-DSV-UNet Architecture
========================
Low-Light image enhancement network with:
    - Residual Dense Block encoder / decoder stages
    - Dual-path downsampling (AveragePool + MaxPool branch from input)
    - Contextual Attention skip connections
    - Multi-Level Fusion Deep Supervision (MLFD-S) — the key innovation

MLFD-S collects intermediate decoder outputs at four scales, projects each to
RGB with tanh activation, aligns them to full resolution, and fuses via
element-wise addition.  Unlike standard deep supervision (which computes a
separate loss per auxiliary head), MLFD-S produces a single fused output that
passes through one unified loss — supervising multi-level reconstruction
implicitly while keeping training simple and stable.
"""

import tensorflow as tf
from tensorflow.keras import layers

from .blocks import RDBNet, decoder_block, upsample, downsample
from .losses import custom_loss


def build_lldsvunet(
    input_shape: tuple = (200, 300, 3),
    base_filters: int = 32,
    learning_rate: float = 1e-5,
    beta_2: float = 0.99,
) -> tf.keras.Model:
    """
    Build and compile the LL-DSV-UNet model.

    Args:
        input_shape    : (H, W, C) — default matches LOL dataset preprocessing.
        base_filters   : Number of filters in the first encoder stage.
                         Subsequent stages double: 32 → 64 → 128 → 256 → 512.
        learning_rate  : Adam learning rate.
        beta_2         : Adam beta_2 parameter.

    Returns:
        tf.keras.Model: Compiled model ready for training.

    Architecture Overview:
        Encoder  : 4 stages  (32 → 64 → 128 → 256 filters)
        Bottleneck: 512 filters, dilation_rate=(5,5)
        Decoder  : 4 stages  (256 → 128 → 64 → 32 filters)
        MLFD-S   : 4 auxiliary RGB heads fused by addition → single output
    """
    inputs = layers.Input(shape=input_shape)

    f = base_filters  # 32

    # -----------------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------------
    # Stage 1  (32 ch)
    x1   = layers.Conv2D(f, (3, 3), padding="same")(inputs)
    x1   = layers.LeakyReLU()(x1)
    x1   = RDBNet(x1, f, (1, 1))
    avg1 = layers.AveragePooling2D((2, 2))(x1)
    e1   = layers.Add()([avg1, downsample(inputs, 1, f)])

    # Stage 2  (64 ch)
    x2   = layers.Conv2D(f * 2, (3, 3), padding="same")(e1)
    x2   = layers.LeakyReLU()(x2)
    x2   = RDBNet(x2, f * 2, (1, 1))
    avg2 = layers.AveragePooling2D((2, 2))(x2)
    e2   = layers.Add()([avg2, downsample(inputs, 2, f * 2)])

    # Stage 3  (128 ch)
    x3   = layers.Conv2D(f * 4, (3, 3), padding="same")(e2)
    x3   = layers.LeakyReLU()(x3)
    x3   = RDBNet(x3, f * 4, (1, 1))
    avg3 = layers.AveragePooling2D((2, 2))(x3)
    e3   = layers.Add()([avg3, downsample(inputs, 3, f * 4)])

    # Stage 4  (256 ch)
    x4   = layers.Conv2D(f * 8, (3, 3), padding="same")(e3)
    x4   = layers.LeakyReLU()(x4)
    x4   = RDBNet(x4, f * 8, (1, 1))
    avg4 = layers.AveragePooling2D((2, 2))(x4)
    e4   = layers.Add()([avg4, downsample(inputs, 4, f * 8)])

    # -----------------------------------------------------------------------
    # Bottleneck  (512 ch, wide dilation)
    # -----------------------------------------------------------------------
    bn = layers.Conv2D(f * 16, (3, 3), padding="same")(e4)
    bn = layers.LeakyReLU()(bn)
    bn = RDBNet(bn, f * 16, (5, 5))

    # -----------------------------------------------------------------------
    # Decoder
    # -----------------------------------------------------------------------
    d1 = layers.Conv2DTranspose(f * 8, (1, 1), strides=(2, 2), padding="same")(bn)
    d1 = decoder_block(d1, x4, f * 8)

    d2 = layers.Conv2DTranspose(f * 4, (1, 1), strides=(2, 2), padding="same")(d1)
    d2 = decoder_block(d2, x3, f * 4)

    d3 = layers.Conv2DTranspose(f * 2, (1, 1), strides=(2, 2), padding="same")(d2)
    d3 = decoder_block(d3, x2, f * 2)

    d4 = layers.Conv2DTranspose(f, (1, 1), strides=(2, 2), padding="same")(d3)
    d4 = decoder_block(d4, x1, f)

    # -----------------------------------------------------------------------
    # Multi-Level Fusion Deep Supervision (MLFD-S)
    # -----------------------------------------------------------------------
    # Each decoder stage produces an auxiliary RGB prediction at its native
    # resolution.  All four are upsampled / aligned to the full output size
    # and fused by element-wise addition before the single loss computation.
    target_h, target_w = d4.shape[1], d4.shape[2]

    ds1 = upsample(d1, 3)
    ds1 = layers.Conv2D(3, (1, 1), padding="same", activation="tanh")(ds1)
    ds1 = layers.Resizing(target_h, target_w)(ds1)

    ds2 = upsample(d2, 2)
    ds2 = layers.Conv2D(3, (1, 1), padding="same", activation="tanh")(ds2)
    ds2 = layers.Resizing(target_h, target_w)(ds2)

    ds3 = upsample(d3, 1)
    ds3 = layers.Conv2D(3, (1, 1), padding="same", activation="tanh")(ds3)
    ds3 = layers.Resizing(target_h, target_w)(ds3)

    ds4 = layers.Conv2D(3, (1, 1), padding="same", activation="tanh")(d4)

    # Single fused output — one loss computation supervises all levels
    output = layers.Add()([ds1, ds2, ds3, ds4])

    # -----------------------------------------------------------------------
    # Compile
    # -----------------------------------------------------------------------
    model = tf.keras.Model(inputs, output, name="LL-DSV-UNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_2=beta_2)
    model.compile(optimizer=optimizer, loss=custom_loss)

    return model
