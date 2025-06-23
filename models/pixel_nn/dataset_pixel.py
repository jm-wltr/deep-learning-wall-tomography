import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
from datetime import datetime

from common.config import *
from common.pmatrix import tensor_pmatrix
from common.waveforms import load_all_waveforms
from common.dmatrix import load_ray_tensor
from models.autoencoder import ConvAutoencoder, DatasetAutoencoder


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
                 autoencoder,
                 nX: int = 30,
                 nY: int = 20,
                 Xmin: float = None,
                 Xmax: float = None,
                 Ymin: float = None,
                 Ymax: float = None,
                 path_waveforms: Path = Path(WAVES_DIR),
                 path_sections: Path = Path(SECTIONS_DIR),
                 path_rays: Path = Path(RAYS_DIR),
                 color_stone=None,
                 color_mortar=None,
                 interpolation=cv2.INTER_LINEAR,
                 binarized: bool = True,
                 skips: list = None,
                 save_tensor: bool = True,
                 save_path: Path = Path(BASE_DIR) / Path("artifacts/pixel_nn"),
                 reduction: str = "resample",
                 reduction_n: int = 200):
        
        # Set default values and skips if not provided
        if color_stone is None:
            color_stone = np.array(colors["Piedra"], dtype=np.float32)
        if color_mortar is None:
            color_mortar = np.array(colors["Mortero"], dtype=np.float32)
        if skips is None:
            skips = []
        if Xmin is None or Xmax is None or Ymin is None or Ymax is None:
            Xmin, Xmax, Ymin, Ymax = dims[0][0], dims[0][1], dims[1][0], dims[1][1]

        self.autoencoder = autoencoder
        self.nX = nX
        self.nY = nY
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Ymin = Ymin
        self.Ymax = Ymax
        self.path_waveforms = path_waveforms
        self.path_sections = path_sections
        self.path_rays = path_rays
        self.color_stone = color_stone
        self.color_mortar = color_mortar
        self.interpolation = interpolation
        self.binarized = binarized
        self.skips = skips
        self.save_tensor = save_tensor
        self.reduction = reduction
        self.reduction_n = reduction_n

        # Encode all waveforms via autoencoder
        wave_ds = DatasetAutoencoder(path=self.path_waveforms,
                                     reduction=self.reduction,
                                     n=self.reduction_n,
                                     save=False,
                                     force_reload=False)
        loader = torch.utils.data.DataLoader(wave_ds, batch_size=128, shuffle=False)
        enc_list = []
        self.autoencoder.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.unsqueeze(1).to(DEVICE)
                enc = self.autoencoder.encode(batch)
                enc_list.append(enc.cpu())
        self.encoded_waveforms = torch.cat(enc_list, dim=0)  # Shape: (num_waveforms, encoding_dim)
        print(f"Encoded waveforms shape: {self.encoded_waveforms.shape} (num_waveforms, encoding_dim)")

        # Load ray tensors for each sample
        section_files = sorted(self.path_rays.glob("ray*.txt")) 
        ray_list = []
        for fpath in section_files:
            # load_ray_tensor returns D of shape (nX, nY, num_rays=66)
            D, _, _ = load_ray_tensor(str(fpath), self.nX, self.nY,
                                    pad=1, include_edges=True)
            # transpose to (num_rays, nX, nY) for PyTorch
            ray_list.append(torch.tensor(D.transpose(2, 0, 1)))
        self.ray_tensors = torch.stack(ray_list).float() # Shape: (num_sections=100, num_rays=66, nX, nY)
        print(f"Ray tensors shape: {self.ray_tensors.shape} (num_samples, num_rays, nX, nY)")

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
        print(f"Loaded {self.num_samples} section files with labels shape: {self.labels.shape} (num_samples, nY, nX)")

        if save_tensor:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ae_name = getattr(self.autoencoder, "model_name", "AE")
            bin_flag = "bin" if self.binarized else "gray"
            
            filename = f"pixel_dataset_{ae_name}_{self.reduction}{self.reduction_n}_grid{self.nX}x{self.nY}_{bin_flag}_{timestamp}.pt"
            save_path_full = save_path / filename

            self.save(save_path_full)

    def __len__(self):
        return self.num_samples * self.nX * self.nY
    
    def __getitem__(self, idx):
        """
        Retrieve the feature vector and label for a given pixel (or batch of pixels).

        Args:
            idx (int or slice): Flat index over all pixels, or a slice for batching.

        Returns:
            If idx is a slice:
                features: Tensor of shape (batch_size, feature_dim)
                labels:   Tensor of shape (batch_size,)
            Else:
                features: Tensor of shape (4 + 66*(1 + enc_dim),)
                label:    Scalar tensor containing the pixel’s ground-truth class/value
        """

        # 1) Batch-slice support: return stacked features & labels
        if isinstance(idx, slice):
            A, B = zip(*[self[j] for j in range(*idx.indices(len(self)))])
            # A: list of feature vectors → stack into (batch_size, feature_dim)
            # B: list of labels → stack into (batch_size,)
            return torch.stack(A), torch.stack(B)

        # 2) Convert flat index → (section p, row y, col x)
        p, y, x = index_to_triplet(idx, self.nX, self.nY)

        # 3) Compute pixel’s real-world bounds: [X0, Y0, X1, Y1]
        limites = pixel_bounds(
            x, y,
            self.nX, self.nY,
            self.Xmin, self.Xmax,
            self.Ymin, self.Ymax
        )  # Tensor shape: (4,)

        # 4) Extract the 66×enc_dim block corresponding to section p
        #    encoded_waveforms viewed as (num_sections, 66, enc_dim)
        enc_block = self.encoded_waveforms.view(self.num_samples, 66, -1)[p]
        # Tensor shape: (66, enc_dim)

        # 5) Get distances for all 66 rays through pixel (y, x)
        ray_dist = self.ray_tensors[p, :, x, y].unsqueeze(1)
        # Tensor shape: (66, 1)

        # 6) Concatenate per-ray distance with that ray’s encoding
        #    Resulting shape: (66, 1 + enc_dim)
        O = torch.cat([ray_dist, enc_block], dim=1)

        # 7) Flatten per-ray features and prepend pixel bounds
        #    Final feature vector shape: (4 + 66*(1 + enc_dim),)
        features = torch.cat([limites, O.flatten()], dim=0)

        # 8) Fetch the ground-truth label for this pixel
        label = torch.tensor(self.labels[p, y, x])
        # Scalar tensor

        return features, label



    # ALTERNATIVE IMPLEMENTATION I WILL HAVE TO TRY LATER
    # def __getitem__(self, idx):
    #     # Support slicing
    #     if isinstance(idx, slice):
    #         return [self[i] for i in range(*idx.indices(len(self)))]
        
    #     # how many rays per section?
    #     n_rays = self.ray_tensors.size(1)  # → 66

    #     # locate the block of 66 encodings for this section
    #     start = sample_idx * n_rays
    #     end   = start + n_rays
    #     enc_block = self.encoded_waveforms[start:end]  # shape: (66, 32)

    #     # Option A: flatten all per‐ray encodings
    #     enc_flat = enc_block.flatten()                 # shape: (66*32,)  

    #     # Option B: average them into one 32‐vector
    #     # enc_flat = enc_block.mean(dim=0)             # shape: (32,)

    #     # ray distances and coords as before
    #     rays   = self.ray_tensors[sample_idx, :, y, x] # (66,)
    #     coords = pixel_bounds(x,y,…)                   # (4,)

    #     # build final feature vector
    #     features = torch.cat([enc_flat, rays, coords], dim=0)

    #     label = torch.tensor(self.labels[sample_idx, y, x])
    #     return features, label

