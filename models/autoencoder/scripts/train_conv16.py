#!/usr/bin/env python3
"""
Train script for ConvAutoencoder16 model with configurable data reduction.

Usage:
    python -m models.autoencoder.scripts.train_conv16 \
        --data-dir data/waves \
        --reduction resample --n 100 \
        --batch-size 32 \
        --lr 1e-3 \
        --epochs 50 \
        --train-frac 0.8 \
        --seed 42 \
        --save-dir artifacts/autoencoder/checkpoints \
        --n 0 \
        --reduction ""
"""
import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.nn import MSELoss

from models.autoencoder.dataset_autoencoder import DatasetAutoencoder
from models.autoencoder.architectures.conv_autoencoder16 import ConvAutoencoder16


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvAutoencoder16 on waveform data with reduction")
    parser.add_argument("--data-dir", type=Path, default="data/waveforms",
                        help="Directory containing raw waveform data")
    parser.add_argument("--reduction", type=str, default="",
                        help="Reduction method: '', 'resample', 'mean', 'max', etc.")
    parser.add_argument("--n", type=int, default=0,
                        help="Parameter for the reduction method (e.g. target length or window size)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction of data to use for training (rest for validation)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--save-dir", type=Path, default="artifacts/autoencoder/checkpoints",
                        help="Directory to save trained model checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess dataset with reduction parameters
    dataset = DatasetAutoencoder(
        path=args.data_dir,
        reduction=args.reduction,
        n=args.n,
        save=True,
        force_reload=False
    )

    # Instantiate model and split data
    model = ConvAutoencoder16(
        dataset=dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_frac=args.train_frac,
        seed=args.seed
    )

    # Set up optimizer and loss criterion
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = MSELoss(reduction='sum')

    # Train model
    model.train_model(
        train_loader=model.train_loader,
        val_loader=model.val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.epochs,
        log_mode='epoch'
    )

    # Save final checkpoint
    ckpt_path = args.save_dir / f"{model.run_name}.pt"
    model.save(ckpt_path)
    print(f"Model checkpoint saved to: {ckpt_path}")


if __name__ == '__main__':
    main()
