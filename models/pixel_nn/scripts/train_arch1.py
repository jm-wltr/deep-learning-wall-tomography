import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from models.autoencoder import DatasetAutoencoder, ConvAutoencoder
from models.pixel_nn import PixelDataset, PixelClassifier
from common.config import DEVICE, BASE_DIR, WAVES_DIR, RAYS_DIR, SECTIONS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pixel-wise classifier on tomography features"
    )
    parser.add_argument(
        "--ae-ckpt", type=Path, required=True,
        help="Path to trained ConvAutoencoder checkpoint"
    )
    parser.add_argument(
        "--ae-reduction", type=str, default="resample",
        help="Data reduction method for AE (resample, mean, etc.)")
    parser.add_argument(
        "--ae-n", type=int, default=200,
        help="Parameter n for reduction in AE")
    parser.add_argument(
        "--waveforms-dir", type=Path, default=Path(WAVES_DIR),
        help="Directory containing raw waveform data"
    )
    parser.add_argument(
        "--sections-dir", type=Path, default=Path(SECTIONS_DIR),
        help="Directory containing cross-section images"
    )
    parser.add_argument(
        "--rays-dir", type=Path, default=Path(RAYS_DIR),
        help="Directory containing ray-distance .txt files"
    )
    parser.add_argument(
        "--nX", type=int, default=30,
        help="Number of pixels in X direction"
    )
    parser.add_argument(
        "--nY", type=int, default=20,
        help="Number of pixels in Y direction"
    )
    parser.add_argument(
        "--binarized", action="store_true",
        help="Use binary (stone/mortar) labels instead of grayscale"
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of non-reserved pixels to use for training"
    )
    parser.add_argument(
        "--n-reserved", type=int, default=5,
        help="Number of entire sections to hold out for final test"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-dir", type=Path,
        default=Path(BASE_DIR) / 'artifacts' / 'pixel_nn' / 'checkpoints',
        help="Directory to save trained model checkpoints"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prepare and load autoencoder
    ae_dataset = DatasetAutoencoder(
        path=args.waveforms_dir,
        reduction=args.ae_reduction,
        n=args.ae_n,
        save=False,
        force_reload=False
    )
    autoencoder = ConvAutoencoder.load(
        path=args.ae_ckpt,
        dataset=ae_dataset,
        device=DEVICE
    ).to(DEVICE)
    autoencoder.eval()
    print(f"Loaded ConvAE: {args.ae_ckpt}")

    # 2) Build PixelDataset with AE encoder
    pixel_dataset = PixelDataset(
        autoencoder=autoencoder,
        nX=args.nX,
        nY=args.nY,
        path_waveforms=args.waveforms_dir,
        path_sections=args.sections_dir,
        path_rays=args.rays_dir,
        binarized=args.binarized,
        save=True,
        reduction=args.ae_reduction,
        reduction_n=args.ae_n
    )

    # 3) Instantiate pixel classifier
    model = PixelClassifier(
        dataset=pixel_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_frac=args.train_frac,
        n_reserved=args.n_reserved,
        seed=args.seed,
        timestamp=True,
        binary=args.binarized
    ).to(DEVICE)
    print(f"Pixel classifier: {model.run_name}")

    # 4) Setup loss and optimizer
    criterion = nn.BCELoss() if args.binarized else nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # 5) Train
    model.train_model(
        train_loader=model.train_loader,
        val_loader=model.val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.epochs,
        log_mode='epoch'
    )

    # 6) Save checkpoint
    ckpt_path = args.save_dir / f"{model.run_name}.pt"
    model.save(ckpt_path)
    print(f"Saved pixel model checkpoint to: {ckpt_path}")


if __name__ == '__main__':
    main()
