from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..autoencoder_base import AutoencoderBase
from ..dataset_autoencoder import DatasetAutoencoder
from ..utils import split_dataset

from common.config import DEVICE

MODEL_NAME = "ConvAE16"
DESCRIPTION = (
    "1D Conv Autoencoder w/ latent dim=16:\n"
    "- Encoder: Conv1d→ELU×2→Flatten→Linear→ELU×3→Linear(→16)\n"
    "- Decoder: mirror linear→Unflatten→ConvTranspose1d×2"
)


class ConvAutoencoder16(AutoencoderBase):
    """
    Concrete 1D convolutional autoencoder with a 16-dimensional latent bottleneck.
    """

    def __init__(
        self,
        dataset: DatasetAutoencoder,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        train_frac: float = 0.8,
        seed: int = None,
        timestamp: bool = True
    ):
        # Initialize base class (handles device, logging, hyperparams)
        super().__init__(
            model_name=MODEL_NAME,
            description=DESCRIPTION,
            batch_size=batch_size,
            learning_rate=learning_rate,
            timestamp=timestamp
        )

        # Split dataset into train and validation subsets
        train_set, val_set = split_dataset(dataset, train_frac, seed)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        # Number of timepoints in each waveform
        length = dataset.length

        # Define encoder: two conv layers followed by linear bottleneck
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=5, padding=5),  # -> (batch,8,length/5)
            nn.ELU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),   # -> (batch,16,length/10)
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16 * (length // 10), 1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Linear(32, 16),
        )

        # Define decoder: mirror of encoder layers
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, 256),
            nn.ELU(),
            nn.Linear(256, 1024),
            nn.ELU(),
            nn.Linear(1024, 16 * (length // 10)),
            nn.ELU(),
            nn.Unflatten(1, (16, length // 10)),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(8, 1, kernel_size=15, stride=5, padding=5),
        )

    def log_reconstruction(self, val_loader, criterion):
        """
        Plot a small batch of original vs. reconstructed waveforms for TensorBoard.
        """
        self.eval()
        # Grab first six examples
        with torch.no_grad():
            batch = next(iter(val_loader))[:6]
            originals = batch.unsqueeze(1).to(self.device)
            reconstructions = self(originals).unsqueeze(1).cpu()

        # Create figure with 2×3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            orig = originals[idx].cpu().squeeze().numpy()
            recon = reconstructions[idx].squeeze().numpy()
            ax.plot(orig, label='Original')
            ax.plot(recon, label='Reconstructed')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize='small')

        fig.suptitle(f"Epoch {self.epochs_trained}")
        fig.tight_layout()
        return fig
