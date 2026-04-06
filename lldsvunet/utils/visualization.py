"""
Visualization utilities for LL-DSV-UNet.

Functions:
    - plot_sample_pairs    : Side-by-side low-light / enhanced / ground-truth grid
    - plot_training_curves : Training and validation loss over epochs
    - save_enhanced_image  : Save a single enhanced image to disk
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, List


def plot_sample_pairs(
    x: np.ndarray,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    num_images: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 5),
) -> None:
    """
    Display a grid of low-light input, enhanced output, and ground truth.

    Args:
        x          : Low-light inputs  [N, H, W, 3].
        y_pred     : Model predictions [N, H, W, 3].
        y_true     : Ground-truth images [N, H, W, 3]. Optional.
        num_images : Number of samples to display.
        save_path  : If given, save figure to this path.
        figsize    : Matplotlib figure size.
    """
    n   = min(num_images, len(x))
    rows = 3 if y_true is not None else 2
    titles = ["Low-light Input", "Enhanced (LL-DSV-UNet)", "Ground Truth"]

    fig, axes = plt.subplots(rows, n, figsize=figsize)
    if n == 1:
        axes = axes.reshape(rows, 1)

    for col in range(n):
        axes[0, col].imshow(np.clip(x[col], 0, 1))
        axes[0, col].set_title(titles[0], fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(np.clip(y_pred[col], 0, 1))
        axes[1, col].set_title(titles[1], fontsize=9)
        axes[1, col].axis("off")

        if y_true is not None:
            axes[2, col].imshow(np.clip(y_true[col], 0, 1))
            axes[2, col].set_title(titles[2], fontsize=9)
            axes[2, col].axis("off")

    plt.suptitle("LL-DSV-UNet Enhancement Results", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {save_path}")

    plt.show()


def plot_training_curves(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    title: str = "LL-DSV-UNet Training Loss",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot training (and optional validation) loss curves.

    Args:
        train_loss : List of per-epoch training loss values.
        val_loss   : List of per-epoch validation loss values. Optional.
        title      : Plot title.
        save_path  : If given, save figure to this path.
        figsize    : Matplotlib figure size.
    """
    plt.figure(figsize=figsize)
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label="Training Loss", linewidth=1.2)
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Validation Loss", linewidth=1.2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {save_path}")

    plt.show()


def save_enhanced_image(image: np.ndarray, save_path: str) -> None:
    """
    Save a single enhanced image (float [0,1]) to disk.

    Args:
        image     : np.ndarray  [H, W, 3] float32 ∈ [0, 1].
        save_path : Output file path (.png recommended).
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(save_path, np.clip(image, 0, 1))
    print(f"Saved → {save_path}")
