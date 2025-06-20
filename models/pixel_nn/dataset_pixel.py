import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset

from common.config import Xmin, Xmax, Ymin, Ymax, SECTIONS_DIR, colors, RAYS_DIR
from common.pmatrix import tensor_pmatrix
from common.waveforms import load_waveform_dataset  # Replace with your waveform loader
from common.dmatrix import load_ray_tensor
from models.autoencoder import ConvAutoencoder

model = ConvAutoencoder()


def index_to_triplet(idx: int, nX: int, nY: int):
    """
    Given a flat index, return the (sample, y, x) triplet.
    """
    sample_idx, rem = divmod(idx, nX * nY)
    y, x = divmod(rem, nX)
    return sample_idx, y, x


def pixel_bounds(i: int, j: int, nX: int, nY: int,
                 Xmin: float, Xmax: float, Ymin: float, Ymax: float) -> torch.Tensor:
    """
    Return bounding box of pixel (i, j) in domain coordinates.
    """
    dX, dY = (Xmax - Xmin) / nX, (Ymax - Ymin) / nY
    X0, X1 = i * dX + Xmin, (i + 1) * dX + Xmin
    Y0, Y1 = j * dY + Ymin, (j + 1) * dY + Ymin
    return torch.tensor([X0, Y0, X1, Y1])


class PixelDataset(Dataset):
    """
    Dataset class for training pixel-wise classifiers.

    - Each sample represents a pixel from a masonry cross-section.
    - Combines ray intersection data, autoencoder-compressed waveforms, and pixel spatial coordinates.
    - Targets are binary or grayscale labels from `Pmatrix`.
    """

    def __init__(self,
                 autoencoder=None,
                 path_numerical: Path = Path("data/numerical analyses"),
                 path_sections: Path = SECTIONS_DIR,
                 path_rays: Path = RAYS_DIR,
                 color_stone=np.array(colors["Piedra"]),
                 color_mortar=np.array(colors["Mortero"]),
                 interpolation=cv2.INTER_LINEAR,
                 binarized: bool = True,
                 nX: int = 30,
                 nY: int = 20,
                 skips: list = [],
                 save_tensor: bool = True):

        self.Xmin, self.Xmax = Xmin, Xmax
        self.Ymin, self.Ymax = Ymin, Ymax
        self.nX, self.nY = nX, nY
        self.binarized = binarized

        # Load and normalize waveform data, then encode with Autoencoder
        raw_waveforms = load_waveform_dataset(path_numerical, autoencoder.reduction, autoencoder.n, skips)
        raw_waveforms = (raw_waveforms - raw_waveforms.mean()) / raw_waveforms.std()

        # Encode waveforms with Autoencoder
        with torch.no_grad():
            encoded = [autoencoder.encode(torch.tensor(w)).detach() for w in raw_waveforms]
            self.encoded_waveforms = torch.stack(encoded).float()

        # Load ray tensors for each sample
        self.ray_tensors = []
        for i in range(100):
            if i in skips:
                continue
            file = path_rays / f"{i:02}.txt"
            ray_tensor, _, _ = load_ray_tensor(str(file), nX, nY, pad=1, include_edges=True)
            self.ray_tensors.append(torch.tensor(ray_tensor.transpose(2, 0, 1)))  # shape: (num_rays, nY, nX)
        self.ray_tensors = torch.stack(self.ray_tensors).float()

        # Load pixel-wise labels (grayscale or binary)
        self.filepaths, self.labels = tensor_pmatrix(
            folder=path_sections,
            nx=nX, ny=nY,
            color_stone=color_stone,
            color_mortar=color_mortar,
            interpolation=interpolation,
            binary=binarized,
            skips=skips
        )

        self.num_samples = len(self.filepaths)
        if save_tensor:
            self.save(Path(f"models/pixel_nn/data/{autoencoder.model_name}.pt"))

    def __len__(self):
        return self.num_samples * self.nX * self.nY

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            items = [self[i] for i in range(*idx.indices(len(self)))]
            r