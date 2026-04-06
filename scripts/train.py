"""
Training script for LL-DSV-UNet.

Usage
-----
    python scripts/train.py --config configs/default.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/default.yaml \\
        --resume checkpoints/LL-DSV-UNet-best.h5

    # Override config values inline
    python scripts/train.py --config configs/default.yaml \\
        --epochs 500 --batch_size 16 --lr 5e-6
"""

import argparse
import os
import sys
import yaml
import numpy as np
import tensorflow as tf

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lldsvunet.model.architecture import build_lldsvunet
from lldsvunet.model.losses import custom_loss
from lldsvunet.data.dataset import load_lol_dataset
from lldsvunet.utils.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train LL-DSV-UNet")
    parser.add_argument("--config",     type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to .h5 checkpoint to resume training from")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override config batch_size")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Override config learning_rate")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.epochs     is not None: config["training"]["epochs"]        = args.epochs
    if args.batch_size is not None: config["training"]["batch_size"]    = args.batch_size
    if args.lr         is not None: config["training"]["learning_rate"] = args.lr

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    seed = config.get("seed", 42)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    data_cfg = config["data"]
    custom_dirs = None
    if data_cfg.get("custom_low") and data_cfg.get("custom_high"):
        custom_dirs = list(zip(data_cfg["custom_low"], data_cfg["custom_high"]))

    x_train, y_train, x_val, y_val, x_eval, y_eval = load_lol_dataset(
        lol_root    = data_cfg["lol_root"],
        custom_dirs = custom_dirs,
        target_size = tuple(data_cfg["target_size"]),
        val_split   = data_cfg.get("val_split", 0.1),
        random_state= seed,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    train_cfg = config["training"]
    model_cfg = config["model"]

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = tf.keras.models.load_model(
            args.resume, custom_objects={"custom_loss": custom_loss}
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_cfg["learning_rate"],
                beta_2=train_cfg.get("beta_2", 0.99),
            ),
            loss=custom_loss,
        )
    else:
        model = build_lldsvunet(
            input_shape   = tuple(data_cfg["target_size"]) + (3,),
            base_filters  = model_cfg.get("base_filters", 32),
            learning_rate = train_cfg["learning_rate"],
            beta_2        = train_cfg.get("beta_2", 0.99),
        )

    model.summary()

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("assets/figures", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = "checkpoints/LL-DSV-UNet-best.h5",
            monitor        = "val_loss",
            save_best_only = True,
            verbose        = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor  = "val_loss",
            patience = train_cfg.get("early_stopping_patience", 100),
            verbose  = 1,
            restore_best_weights = True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = train_cfg.get("reduce_lr_patience", 50),
            min_lr   = 1e-7,
            verbose  = 1,
        ),
        tf.keras.callbacks.CSVLogger("checkpoints/training_log.csv"),
    ]

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    history = model.fit(
        x_train, y_train,
        batch_size      = train_cfg["batch_size"],
        epochs          = train_cfg["epochs"],
        validation_data = (x_val, y_val),
        callbacks       = callbacks,
    )

    # -----------------------------------------------------------------------
    # Save final model & history
    # -----------------------------------------------------------------------
    model.save("checkpoints/LL-DSV-UNet-final.h5")
    np.save("checkpoints/train_loss.npy", np.array(history.history["loss"]))
    np.save("checkpoints/val_loss.npy",   np.array(history.history["val_loss"]))

    plot_training_curves(
        train_loss = history.history["loss"],
        val_loss   = history.history["val_loss"],
        save_path  = "assets/figures/training_curves.png",
    )

    print("\nTraining complete.")
    print("  Best checkpoint : checkpoints/LL-DSV-UNet-best.h5")
    print("  Final model     : checkpoints/LL-DSV-UNet-final.h5")


if __name__ == "__main__":
    main()
