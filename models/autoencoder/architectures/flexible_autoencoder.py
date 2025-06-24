from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from ..autoencoder_base import AutoencoderBase
from ..dataset_autoencoder import DatasetAutoencoder
from ..utils import split_dataset
from common.config import DEVICE

class ConvAutoencoder(AutoencoderBase):
    """
    1D Convolutional Autoencoder with dynamic naming and description.
    """
    def __init__(
        self,
        dataset,
        latent_dim: int = 16,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        train_frac: float = 0.8,
        seed: int = None,
        timestamp: bool = True,
        reduction: str = "",
        reduction_n: int = 0
    ):
        # Store config
        super().__init__(
            model_name=self._build_name(latent_dim, dropout, use_batchnorm, reduction, reduction_n),
            description=self._build_description(latent_dim, dropout, use_batchnorm),
            batch_size=batch_size,
            learning_rate=learning_rate,
            timestamp=timestamp
        )
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.train_frac = train_frac
        self.seed = seed
        self.reduction = reduction
        self.reduction_n = reduction_n

        enc_cfg = dict(channels=[1, 8, 16], kernels=[15, 5], strides=[5, 2])
        from ..utils import split_dataset
        train_set, val_set = split_dataset(dataset, train_frac, seed)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

        length = dataset.length
        channels, kernels, strides = enc_cfg['channels'], enc_cfg['kernels'], enc_cfg['strides']
        down_factor = strides[0] * strides[1]
        flat_size = channels[-1] * (length // down_factor)

        # Build encoder
        enc_layers = []
        for i in range(len(strides)):
            enc_layers.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernels[i], stride=strides[i], padding=kernels[i]//2))
            if use_batchnorm: enc_layers.append(nn.BatchNorm1d(channels[i+1]))
            enc_layers.append(nn.ELU())
            if dropout > 0: enc_layers.append(nn.Dropout(dropout))
        enc_layers.append(nn.Flatten())
        enc_layers.extend([nn.Linear(flat_size, 1024), nn.ELU()] + ([nn.Dropout(dropout)] if dropout>0 else []) + [nn.Linear(1024, latent_dim)])
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder
        dec_layers = [nn.Linear(latent_dim, 1024), nn.ELU()] + ([nn.Dropout(dropout)] if dropout>0 else []) + [nn.Linear(1024, flat_size), nn.ELU(), nn.Unflatten(1, (channels[-1], length//down_factor))]
        for i in reversed(range(len(strides))):
            out_ch = channels[i] if i > 0 else 1
            dec_layers.append(nn.ConvTranspose1d(channels[i+1], out_ch, kernel_size=kernels[i], stride=strides[i], padding=kernels[i]//2, output_padding=strides[i]-1))
            if i > 0:
                if use_batchnorm: dec_layers.append(nn.BatchNorm1d(out_ch))
                dec_layers.append(nn.ELU())
                if dropout>0: dec_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec_layers)

    @staticmethod
    def _build_name(latent_dim, dropout, use_batchnorm, reduction, reduction_n):
        parts = ["ConvAE"] + ([f"{reduction}{reduction_n}"] if reduction else []) + [f"lat{latent_dim}"] + ([f"do{int(dropout*100)}"] if dropout>0 else []) + (["bn"] if use_batchnorm else [])
        return "_".join(parts)

    @staticmethod
    def _build_description(latent_dim, dropout, use_batchnorm):
        return (
            f"1D ConvAE with latent={latent_dim}; dropout={dropout}; batchnorm={use_batchnorm}. "
            f"Symmetric encoder/decoder with Conv1d+ELU layers."
        )

    def save(self, path: Path) -> None:
        checkpoint = {
            'state_dict': self.state_dict(),
            'epochs_trained': self.epochs_trained,
            'history': self.history,
            'latent_dim': self.latent_dim,
            'dropout': self.dropout,
            'use_batchnorm': self.use_batchnorm,
            'batch_size': self.batch_size,
            'learning_rate': self.lr,
            'train_frac': self.train_frac,
            'seed': self.seed,
            'reduction': self.reduction,
            'reduction_n': self.reduction_n,
            'run_name': self.run_name,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, dataset, device=None) -> 'ConvAutoencoder':
        checkpoint = torch.load(path, map_location=device or DEVICE)
        model = cls(
            dataset=dataset,
            latent_dim=checkpoint['latent_dim'],
            dropout=checkpoint['dropout'],
            use_batchnorm=checkpoint['use_batchnorm'],
            batch_size=checkpoint['batch_size'],
            learning_rate=checkpoint['learning_rate'],
            train_frac=checkpoint['train_frac'],
            seed=checkpoint['seed'],
            timestamp=False,
            reduction=checkpoint.get('reduction', ''),
            reduction_n=checkpoint.get('reduction_n', 0)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.epochs_trained = checkpoint['epochs_trained']
        model.history = checkpoint['history']
        model.run_name = checkpoint.get('run_name', model.run_name)
        return model

    def forward(self, x: Tensor) -> Tensor:
        x = self._ensure_batch_and_channel(x)
        return self.decoder(self.encoder(x.to(self.device))).squeeze(1)

    def log_reconstruction(self, val_loader, criterion):
        import matplotlib.pyplot as plt
        self.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))[:6].unsqueeze(1).to(self.device)
            recon = self(batch).unsqueeze(1).cpu()
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for ax, orig, rec in zip(axes.flatten(), batch.cpu(), recon):
            ax.plot(orig.squeeze().numpy(), label='Orig')
            ax.plot(rec.squeeze().numpy(), label='Recon')
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Epoch {self.epochs_trained}")
        return fig
