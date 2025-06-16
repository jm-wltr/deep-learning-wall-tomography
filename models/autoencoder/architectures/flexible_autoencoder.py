"""
conv_autoencoder.py

Configurable 1D convolutional autoencoder that supports:
  - Variable latent dimension
  - Optional dropout
  - Optional batch normalization
  - Custom encoder/decoder layer sizes
Usage:
    model = ConvAutoencoder(
        dataset,
        latent_dim=32,
        dropout=0.2,
        use_batchnorm=True,
        batch_size=64,
        learning_rate=1e-3,
        train_frac=0.8,
        seed=42
    )
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..autoencoder_base import AutoencoderBase
from ..dataset_autoencoder import DatasetAutoencoder
from ..utils import split_dataset

MODEL_NAME = "ConvAutoencoder"
DESCRIPTION = (
    "Configurable 1D Conv Autoencoder with optional dropout and batchnorm,"
    " variable latent size."
)

class ConvAutoencoder(AutoencoderBase):
    """
    1D Convolutional Autoencoder with configurable options:
      - latent_dim: size of bottleneck
      - dropout: dropout probability (0=no dropout)
      - use_batchnorm: whether to insert BatchNorm1d layers
    """
    def __init__(
        self,
        dataset: DatasetAutoencoder,
        latent_dim: int = 16,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        train_frac: float = 0.8,
        seed: int = None,
        timestamp: bool = True
    ):
        super().__init__(model_name=MODEL_NAME,
                         description=DESCRIPTION,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         timestamp=timestamp)

        # Data split
        train_set, val_set = split_dataset(dataset, train_frac, seed)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_set,   batch_size=self.batch_size, shuffle=False)

        length = dataset.length  # waveform length
        # Encoder conv dimensions
        channels = [1, 8, 16]
        strides = [5, 2]
        kernels = [15, 5]

        enc_layers = []
        for i in range(len(strides)):
            enc_layers.append(nn.Conv1d(channels[i], channels[i+1],
                                        kernel_size=kernels[i],
                                        stride=strides[i],
                                        padding=kernels[i]//2))
            if use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(channels[i+1]))
            enc_layers.append(nn.ELU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
        enc_layers.append(nn.Flatten())
        # Compute flattened size
        downsample_factor = strides[0] * strides[1]
        flat_size = channels[-1] * (length // downsample_factor)
        # Linear bottleneck
        enc_layers.extend([
            nn.Linear(flat_size, 1024),
            nn.ELU(),
            nn.Dropout(dropout) if dropout>0 else nn.Identity(),
            nn.Linear(1024, latent_dim)
        ])
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = [nn.Linear(latent_dim, 1024), nn.ELU()]
        if dropout > 0:
            dec_layers.append(nn.Dropout(dropout))
        dec_layers.extend([
            nn.Linear(1024, flat_size),
            nn.ELU()
        ])
        # Unflatten to conv shape
        dec_layers.append(nn.Unflatten(1, (channels[-1], length // downsample_factor)))
        # Transposed convs mirror encoder
        for i in reversed(range(len(strides))):
            out_ch = channels[i] if i>0 else 1
            dec_layers.append(nn.ConvTranspose1d(
                channels[i+1], out_ch,
                kernel_size=kernels[i],
                stride=strides[i],
                padding=kernels[i]//2,
                output_padding=(strides[i]-1)
            ))
            if i>0:
                if use_batchnorm:
                    dec_layers.append(nn.BatchNorm1d(out_ch))
                dec_layers.append(nn.ELU())
                if dropout > 0:
                    dec_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_batch_and_channel(x)
        z = self.encoder(x.to(self.device))
        out = self.decoder(z)
        return out.squeeze(1)

    def log_reconstruction(self, val_loader, criterion):
        import matplotlib.pyplot as plt
        self.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))[:6].unsqueeze(1).to(self.device)
            recon = self(batch).unsqueeze(1).cpu()
        fig, axes = plt.subplots(2, 3, figsize=(12,6))
        for ax, orig, rec in zip(axes.flatten(), batch.cpu(), recon):
            ax.plot(orig.squeeze().numpy(), label='Orig')
            ax.plot(rec.squeeze().numpy(), label='Recon')
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Epoch {self.epochs_trained}")
        return fig
