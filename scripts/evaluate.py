"""
Evaluation script for LL-DSV-UNet.

Computes PSNR, SSIM, and LPIPS on a paired evaluation dataset
and prints a per-image table plus aggregated statistics.

Usage
-----
    python scripts/evaluate.py \\
        --model  checkpoints/LL-DSV-UNet-final.h5 \\
        --low    data/lol_dataset/eval15/low \\
        --high   data/lol_dataset/eval15/high

    # Skip LPIPS (faster, no PyTorch needed)
    python scripts/evaluate.py \\
        --model  checkpoints/LL-DSV-UNet-final.h5 \\
        --low    data/lol_dataset/eval15/low \\
        --high   data/lol_dataset/eval15/high \\
        --no_lpips

    # Save visual comparison grid
    python scripts/evaluate.py \\
        --model     checkpoints/LL-DSV-UNet-final.h5 \\
        --low       data/lol_dataset/eval15/low \\
        --high      data/lol_dataset/eval15/high \\
        --save_grid assets/results/eval_grid.png
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lldsvunet.model.losses import custom_loss
from lldsvunet.data.dataset import load_images_from_directory
from lldsvunet.utils.metrics import compute_psnr, compute_ssim, compute_lpips
from lldsvunet.utils.visualization import plot_sample_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LL-DSV-UNet")
    parser.add_argument("--model",       required=True, help="Path to .h5 model")
    parser.add_argument("--low",         required=True, help="Directory of low-light input images")
    parser.add_argument("--high",        required=True, help="Directory of ground-truth images")
    parser.add_argument("--target_size", nargs=2, type=int, default=[200, 300],
                        metavar=("H", "W"))
    parser.add_argument("--no_lpips",    action="store_true", help="Skip LPIPS computation")
    parser.add_argument("--save_grid",   type=str, default=None,
                        help="Save visual grid to this path")
    parser.add_argument("--num_vis",     type=int, default=5,
                        help="Number of images to show in the visual grid")
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(
        args.model, custom_objects={"custom_loss": custom_loss}
    )

    # -----------------------------------------------------------------------
    # Load evaluation data
    # -----------------------------------------------------------------------
    target_size = tuple(args.target_size)
    print(f"\nLoading evaluation images (resize → {target_size})...")
    x_eval = load_images_from_directory([args.low],  target_size)
    y_eval = load_images_from_directory([args.high], target_size)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    print(f"\nRunning inference on {len(x_eval)} images...")
    y_pred = model.predict(x_eval, verbose=1)

    # -----------------------------------------------------------------------
    # Per-image metrics
    # -----------------------------------------------------------------------
    psnr_scores, ssim_scores = [], []

    header = f"{'#':>3}  {'PSNR (dB)':>10}  {'SSIM':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for i in range(len(y_eval)):
        p = compute_psnr(y_eval[i], y_pred[i])
        s = compute_ssim(y_eval[i], y_pred[i])
        psnr_scores.append(p)
        ssim_scores.append(s)
        print(f"{i+1:>3}  {p:>10.2f}  {s:>8.4f}")

    # -----------------------------------------------------------------------
    # Aggregate stats
    # -----------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(f"  Avg PSNR : {np.mean(psnr_scores):.2f} dB")
    print(f"  Avg SSIM : {np.mean(ssim_scores):.4f}")

    if not args.no_lpips:
        print("  Computing LPIPS (VGG)...")
        lpips_score = compute_lpips(y_eval, y_pred)
        print(f"  Avg LPIPS: {lpips_score:.4f}")
    print("=" * 40)

    # -----------------------------------------------------------------------
    # Visual comparison grid
    # -----------------------------------------------------------------------
    if args.save_grid or args.num_vis > 0:
        plot_sample_pairs(
            x_eval, y_pred, y_eval,
            num_images = args.num_vis,
            save_path  = args.save_grid,
        )


if __name__ == "__main__":
    main()
