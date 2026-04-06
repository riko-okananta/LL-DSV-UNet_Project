"""
Evaluation metrics for Low-Light Image Enhancement.

Metrics implemented:
    - PSNR  : Peak Signal-to-Noise Ratio     (↑ better)
    - SSIM  : Structural Similarity Index    (↑ better)
    - LPIPS : Learned Perceptual Image Patch Similarity (↓ better, VGG backbone)

Training convergence analytics:
    - convergence_score_rdp : Relative Decrease Per epoch
    - stability_index       : Coefficient of Variation against moving average
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim
from skimage.color import rgb2gray
from typing import Dict, List


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------

def compute_psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    PSNR on grayscale images.

    Args:
        y_true : np.ndarray  Ground-truth image [H, W, 3], float32 ∈ [0, 1].
        y_pred : np.ndarray  Predicted image    [H, W, 3], float32 ∈ [0, 1].

    Returns:
        float: PSNR value in dB.
    """
    return _psnr(rgb2gray(y_true), rgb2gray(y_pred), data_range=1.0)


def compute_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    SSIM on grayscale images.

    Args:
        y_true : np.ndarray  Ground-truth image [H, W, 3], float32 ∈ [0, 1].
        y_pred : np.ndarray  Predicted image    [H, W, 3], float32 ∈ [0, 1].

    Returns:
        float: SSIM value ∈ [-1, 1].
    """
    return _ssim(rgb2gray(y_true), rgb2gray(y_pred), data_range=1.0)


def compute_lpips(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    LPIPS with VGG backbone (requires torchmetrics + torch).

    Input arrays are batch arrays: [B, H, W, 3], float ∈ [0, 1].
    Internally clipped to [-1, 1] and converted to PyTorch tensors.

    Args:
        y_true : np.ndarray  Ground-truth batch [B, H, W, 3].
        y_pred : np.ndarray  Predicted batch    [B, H, W, 3].

    Returns:
        float: Mean LPIPS score (lower = more perceptually similar).
    """
    try:
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

        pred_t   = torch.from_numpy(np.clip(y_pred, -1, 1)).permute(0, 3, 1, 2).float()
        target_t = torch.from_numpy(np.clip(y_true, -1, 1)).permute(0, 3, 1, 2).float()

        return float(lpips_fn(target_t, pred_t).numpy())
    except ImportError:
        raise ImportError(
            "LPIPS requires PyTorch and torchmetrics.\n"
            "Install with: pip install torch torchmetrics"
        )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    compute_lpips_score: bool = True,
) -> Dict[str, float]:
    """
    Compute PSNR, SSIM, and optionally LPIPS for a full evaluation set.

    Args:
        y_true              : np.ndarray  [N, H, W, 3]
        y_pred              : np.ndarray  [N, H, W, 3]
        compute_lpips_score : bool        Set False to skip LPIPS (faster).

    Returns:
        dict with keys: "psnr", "ssim", and optionally "lpips".

    Example:
        >>> scores = evaluate_model(y_eval, y_pred)
        >>> print(f"PSNR: {scores['psnr']:.2f} dB | SSIM: {scores['ssim']:.4f}")
    """
    psnr_scores: List[float] = []
    ssim_scores: List[float] = []

    for i in range(len(y_true)):
        psnr_scores.append(compute_psnr(y_true[i], y_pred[i]))
        ssim_scores.append(compute_ssim(y_true[i], y_pred[i]))

    results = {
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
    }

    if compute_lpips_score:
        results["lpips"] = compute_lpips(y_true, y_pred)

    return results


# ---------------------------------------------------------------------------
# Training convergence analytics
# ---------------------------------------------------------------------------

def convergence_score_rdp(loss_values: np.ndarray) -> float:
    """
    Relative Decrease Per epoch (RDP) — measures how efficiently a model
    converges over a training run.

    RDP = (L₀ − L_end) / (L₀ × T)

    Higher RDP → faster convergence relative to initial loss.

    Args:
        loss_values : 1-D array of training or validation loss values.

    Returns:
        float: RDP score.
    """
    loss_values = np.asarray(loss_values)
    L0, L_end, T = loss_values[0], loss_values[-1], len(loss_values)
    return float((L0 - L_end) / (L0 * T))


def stability_index(loss_values: np.ndarray, window_size: int = 20) -> float:
    """
    Stability Index (SI) — coefficient of variation of the residual between
    the loss curve and its moving average.

    Lower SI → more stable training dynamics.

    Args:
        loss_values : 1-D array of loss values.
        window_size : Moving average window (default 20 epochs).

    Returns:
        float: SI value.
    """
    loss_values = np.asarray(loss_values)
    mov_avg = np.convolve(loss_values, np.ones(window_size) / window_size, mode="valid")
    aligned = loss_values[window_size - 1:]
    diffs   = aligned - mov_avg
    return float(np.std(diffs) / np.mean(mov_avg))
