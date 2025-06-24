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
    if isinstance(idx, torch.Tensor) and idx.dim() == 0:
        idx = idx.item()
        
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
    def __init__(
        self,
        autoencoder: ConvAutoencoder,
        nX: int = 30,
        nY: int = 20,
        Xmin: float = None,
        Xmax: float = None,
        Ymin: float = None,
        Ymax: float = None,
        path_waveforms: Path = Path(WAVES_DIR),
        path_sections: Path = Path(SECTIONS_DIR),
        path_rays: Path = Path(RAYS_DIR),
        binarized: bool = True,
        skips: list = [],
        save: bool = True,
        force_reload: bool = False,
        reduction: str = "resample",
        reduction_n: int = 200
    ):
        # assign params
        self.autoencoder   = autoencoder
        self.nX, self.nY   = nX, nY
        self.binarized     = binarized
        self.skips         = skips or []
        self.reduction     = reduction
        self.reduction_n   = reduction_n
        # bounding box defaults
        if Xmin is None or Xmax is None:
            Xmin, Xmax = dims[0]
        if Ymin is None or Ymax is None:
            Ymin, Ymax = dims[1]
        self.Xmin, self.Xmax = Xmin, Xmax
        self.Ymin, self.Ymax = Ymin, Ymax
        # paths
        self.path_waveforms = path_waveforms
        self.path_sections  = path_sections
        self.path_rays      = path_rays

        # prepare cache
        cache_dir = Path(BASE_DIR) / "artifacts" / "pixel_nn"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = f"pixel_{getattr(autoencoder,'model_name','AE')}_{reduction}{reduction_n}_{nX}x{nY}_{'bin' if binarized else 'gray'}.pt"
        cache_path = cache_dir / fname

        # load or build
        if cache_path.exists() and not force_reload:
            payload = torch.load(cache_path)
            self.encoded_waveforms = payload['encoded_waveforms']
            self.ray_tensors       = payload['ray_tensors']
            self.labels            = payload['labels']
            self.num_sections      = payload['num_sections']
        else:
            # 1) encode waveforms
            wave_ds = DatasetAutoencoder(
                path=path_waveforms,
                reduction=reduction,
                n=reduction_n,
                save=False,
                force_reload=False
            )
            loader = torch.utils.data.DataLoader(wave_ds, batch_size=128, shuffle=False)
            enc_list = []
            autoencoder.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = batch.unsqueeze(1).to(DEVICE)
                    enc = autoencoder.encode(batch)
                    enc_list.append(enc.cpu())
            self.encoded_waveforms = torch.cat(enc_list, dim=0)
            print(f"Encoded waveforms with shape {self.encoded_waveforms.shape} (num_waveforms, encoding_dims)")

            # 2) load ray tensors (per section)
            files = sorted(path_rays.glob("ray*.txt"))
            print("Sorted files in rays")
            ray_list = []
            for f in files:
                D, _, _ = load_ray_tensor(str(f), nX, nY, pad=1, include_edges=True)
                ray_list.append(torch.tensor(D.transpose(2,0,1)))
            self.ray_tensors = torch.stack(ray_list).float()
            print(f"Saved ray tensors shape: {self.ray_tensors.shape} (num_sections=100, num_rays=66, nX, nY)")

            # 3) pixel labels
            _, self.labels = tensor_pmatrix(
                folder=path_sections,
                nx=nX, ny=nY,
                binary=binarized,
                skips=skips
            )
            self.num_sections = len(self.labels)
            print(f"Loaded {self.num_sections} section files with labels shape: {self.labels.shape} (num_sections, nY, nX)")

            # save if requested
            if save:
                torch.save({
                    'encoded_waveforms': self.encoded_waveforms,
                    'ray_tensors':       self.ray_tensors,
                    'labels':            self.labels,
                    'num_sections':      self.num_sections
                }, cache_path)


    def __len__(self):
        return self.num_sections * self.nX * self.nY
    
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
        enc_block = self.encoded_waveforms.view(self.num_sections, 66, -1)[p]
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

