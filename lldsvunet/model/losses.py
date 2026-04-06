"""
Custom Loss Functions for LL-DSV-UNet
======================================
Composite loss combining three complementary objectives:

    L_total = L_mae + L_color + 0.2 * L_cs

    L_mae   : Mean Absolute Error — pixel-level reconstruction fidelity.
    L_color : Per-channel MSE + variance of squared residuals — encourages
              color balance and penalises high-variance channel errors.
    L_cs    : Contrast-Structure loss derived from the contrast (C) and
              structure (S) components of SSIM, computed on grayscale images
              using local statistics via avg_pool2d (window=10).

All functions are TensorFlow-compatible and can be serialised for model saving.
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# Component losses
# ---------------------------------------------------------------------------

def contrast_structure_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Contrast-Structure (CS) loss — grayscale, local-statistics based.

    Derived from the C·S terms of SSIM (luminance term excluded).
    Uses a 10×10 pooling window with VALID padding.

    Args:
        y_true : tf.Tensor  Ground-truth images  [B, H, W, 3], range [0, 1].
        y_pred : tf.Tensor  Predicted images      [B, H, W, 3], range [0, 1].

    Returns:
        tf.Tensor: Scalar loss value ∈ [0, 2].
    """
    y_true = tf.image.rgb_to_grayscale(y_true)
    y_pred = tf.image.rgb_to_grayscale(y_pred)

    C2 = 1e-7
    C3 = C2 / 2

    mu_t = tf.nn.avg_pool2d(y_true, ksize=10, strides=1, padding="VALID")
    mu_p = tf.nn.avg_pool2d(y_pred, ksize=10, strides=1, padding="VALID")

    sig_t  = tf.math.sqrt(tf.abs(
        tf.nn.avg_pool2d(tf.square(y_true), ksize=10, strides=1, padding="VALID")
        - tf.square(mu_t)
    ))
    sig_p  = tf.math.sqrt(tf.abs(
        tf.nn.avg_pool2d(tf.square(y_pred), ksize=10, strides=1, padding="VALID")
        - tf.square(mu_p)
    ))
    sig_tp = (
        tf.nn.avg_pool2d(y_true * y_pred, ksize=10, strides=1, padding="VALID")
        - mu_t * mu_p
    )

    contrast  = (2 * sig_t * sig_p + C2) / (tf.square(sig_t) + tf.square(sig_p) + C2)
    structure = (sig_tp + C3) / (sig_t * sig_p + C3)

    return 1.0 - tf.reduce_mean(contrast * structure)


def mae_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Mean Absolute Error (L1) loss.

    Args:
        y_true : tf.Tensor  Ground-truth images.
        y_pred : tf.Tensor  Predicted images.

    Returns:
        tf.Tensor: Scalar MAE value.
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def color_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Per-channel colour fidelity loss.

    Combines MSE and variance of squared residuals for each RGB channel,
    penalising both mean colour shift and channel-wise inconsistency.

    Args:
        y_true : tf.Tensor  Ground-truth images [B, H, W, 3].
        y_pred : tf.Tensor  Predicted images    [B, H, W, 3].

    Returns:
        tf.Tensor: Scalar colour loss.
    """
    total = 0.0
    for c in range(3):
        r_t = tf.reshape(y_true[..., c], [-1])
        r_p = tf.reshape(y_pred[..., c], [-1])
        sq  = tf.square(r_t - r_p)
        total += tf.reduce_mean(sq) + tf.square(tf.math.reduce_variance(sq))
    return total


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Composite LL-DSV-UNet loss:  L_mae + L_color + 0.2 * L_cs

    This function is registered as the model's training objective.
    It must be passed as a custom object when loading a saved model:

        model = tf.keras.models.load_model(
            "LL-DSV-UNet.h5",
            custom_objects={"custom_loss": custom_loss}
        )

    Args:
        y_true : tf.Tensor  Ground-truth images.
        y_pred : tf.Tensor  Predicted images.

    Returns:
        tf.Tensor: Scalar composite loss.
    """
    return mae_loss(y_true, y_pred) + color_loss(y_true, y_pred) + 0.2 * contrast_structure_loss(y_true, y_pred)
