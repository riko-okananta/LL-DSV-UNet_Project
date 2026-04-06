"""
Batch image enhancement script for LL-DSV-UNet.

Processes all images in an input directory and saves enhanced results
to an output directory.

Usage
-----
    # Enhance all images in a folder (auto-resize to training resolution)
    python scripts/enhance.py \\
        --model checkpoints/LL-DSV-UNet-final.h5 \\
        --input  path/to/low_light_images/ \\
        --output path/to/enhanced_output/

    # Keep original resolution (no resize)
    python scripts/enhance.py \\
        --model checkpoints/LL-DSV-UNet-final.h5 \\
        --input  path/to/images/ \\
        --output path/to/output/ \\
        --no_resize
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lldsvunet.model.losses import custom_loss


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Low-Light Enhancement with LL-DSV-UNet")
    parser.add_argument("--model",     required=True,  help="Path to .h5 model file")
    parser.add_argument("--input",     required=True,  help="Input directory with low-light images")
    parser.add_argument("--output",    required=True,  help="Output directory for enhanced images")
    parser.add_argument("--target_size", nargs=2, type=int, default=[200, 300],
                        metavar=("H", "W"), help="Resize images before inference (default: 200 300)")
    parser.add_argument("--no_resize", action="store_true",
                        help="Skip resizing — use original image dimensions")
    parser.add_argument("--ext",       default=".png",
                        help="Output file extension (default: .png)")
    return parser.parse_args()


def load_image(path: str, target_size=None) -> np.ndarray:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    if target_size is not None:
        img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return np.expand_dims(img.numpy(), axis=0)


def main():
    args = parse_args()

    # Load model
    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(
        args.model, custom_objects={"custom_loss": custom_loss}
    )
    print("Model loaded successfully.\n")

    # Gather images
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ])

    if not image_paths:
        print(f"No supported images found in: {input_dir}")
        return

    target_size = None if args.no_resize else tuple(args.target_size)
    print(f"Enhancing {len(image_paths)} images → {output_dir}")
    if target_size:
        print(f"  Resize: {target_size[0]}×{target_size[1]}")
    else:
        print("  Resize: disabled (original resolution)")
    print()

    for img_path in tqdm(image_paths):
        img_batch = load_image(str(img_path), target_size)
        enhanced  = model.predict(img_batch, verbose=0)[0]
        enhanced  = np.clip(enhanced, 0, 1)

        out_name = img_path.stem + args.ext
        out_path = output_dir / out_name
        plt.imsave(str(out_path), enhanced)

    print(f"\nDone! Enhanced images saved to: {output_dir}")


if __name__ == "__main__":
    main()
