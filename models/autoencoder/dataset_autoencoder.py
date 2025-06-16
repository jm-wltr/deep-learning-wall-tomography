# models/autoencoder/dataset.py
"""
Autoencoder dataset loader for raw waveform data.

This module defines a PyTorch Dataset that:
  - Loads preprocessed waveform tensors from a .pt file if given.
  - Otherwise reads all waveform files from WAVES_DIR,
    applies a specified data reduction, normalizes the data,
    and optionally saves the result for faster future loading.
  - Provides unsupervised access to waveforms (no labels).
  - Offers a public method to plot random examples for inspection.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

from common.config import WAVES_DIR, BASE_DIR
from common.waveforms import load_all_waveforms


class DatasetAutoencoder(Dataset):
    def __init__(self,
                 path: Path = Path(WAVES_DIR),
                 reduction: str = "",
                 n: int = 0,
                 save: bool = True,
                 force_reload: bool = False):
        """
        PyTorch Dataset for autoencoder training on waveform data.

        Args:
            path (Path): Path to a directory of raw waves (default WAVES_DIR).
            reduction (str): Reduction method ('resample', 'mean', 'max', or '').
            n (int): Reduction parameter (e.g. number of points or window size).
            save (bool): Whether to save processed tensor to disk.
            force_reload (bool): Whether to force a rebuild from raw data.

        Public Methods:
            plot_samples(n): Plot n random waveforms for inspection.
        """
        # Determine save directory and filename
        self.waves_dir = path
        cache_dir  = Path(BASE_DIR) / "artifacts" / "autoencoder"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = "original.pt" if reduction=="" else f"{reduction}_{n}.pt"
        cache_path = cache_dir / fname

        # Load or create dataset
        if cache_path.exists() and not force_reload:
            # Load pre-saved tensor
            self.data = torch.load(cache_path)
        else:
            # Read raw waveforms and apply reduction
            self.data = load_all_waveforms(path, reduction=reduction, n=n)
            # Normalize to zero-mean, unit-variance
            mean = self.data.mean()
            std = self.data.std()
            self.data = (self.data - mean) / std
            if save:
                torch.save(self.data, cache_path)

        # Record dataset size (num samples x num points per waveform)
        self.num_samples, self.length = self.data.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Unsupervised: return waveform only
        return self.data[idx]

    def plot_samples(self, n: int = 3):
        """
        Plot n random waveforms from the dataset for inspection.

        Args:
            n (int): Number of random samples to plot.
        """
        indices = np.random.choice(self.num_samples, size=n, replace=False)
        plt.figure(figsize=(12, 3))
        for i in indices:
            waveform = self.data[i]
            # If tensor, convert to numpy
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            plt.plot(waveform)
        plt.title(f"Autoencoder dataset: {self.num_samples} samples, length={self.length}")
        plt.show()
