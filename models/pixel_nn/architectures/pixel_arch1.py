from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split

from common.config import DEVICE
from ..pixel_base import PixelBase
from ..dataset_pixel import PixelDataset


def separacion_dataset_supervised(dataset, train_frac=0.8, n_reserved=5, seed=None):
    """
    Split PixelDataset into train/val by pixel, with a set of n_reserved entire sections reserved for testing.
    """
    total_pixels = len(dataset)
    n_sections = dataset.num_sections
    DIM = dataset.nX * dataset.nY
    # choose reserved section indices
    if seed is not None:
        torch.manual_seed(seed)
    reserved = torch.randperm(n_sections)[:n_reserved]

    mask = torch.ones(total_pixels, dtype=torch.bool)
    for sec in reserved:
        mask[sec * DIM:(sec + 1) * DIM] = False

    reserved_subset = Subset(dataset, (~mask).nonzero(as_tuple=True)[0])
    remaining_subset = Subset(dataset, mask.nonzero(as_tuple=True)[0])

    n_train = int(len(dataset) * train_frac)
    n_val_remain = len(remaining_subset) - n_train
    train_subset, val_remain = random_split(remaining_subset, [n_train, n_val_remain])
    val_subset = ConcatDataset([val_remain, reserved_subset])
    return train_subset, val_subset, reserved


class PixelClassifier(PixelBase):
    """
    Fully-connected pixel-wise classifier:
      input_dim -> 512 -> 128 -> 32 -> 1
    """
    def __init__(
        self,
        dataset: PixelDataset,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        train_frac: float = 0.8,
        n_reserved: int = 5,
        seed: int = None,
        timestamp: bool = True,
        binary: bool = True
    ):
        # Build a descriptive name and description
        name = f"PixelMLP_{batch_size}bs_{learning_rate:.0e}lr"
        desc = (
            "Pixel MLP: [input  512  128  32  1], ``ELU`` activations, "
            f"binary={binary}, reserved_sections={n_reserved}"
        )
        super().__init__(
            model_name=name,
            description=desc,
            batch_size=batch_size,
            learning_rate=learning_rate,
            timestamp=timestamp,
            binary=binary
        )

        self.dataset = dataset
        # Split dataset
        train_ds, val_ds, self.reserved_sections = separacion_dataset_supervised(
            dataset, train_frac, n_reserved, seed
        )
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Model layers
        input_dim = len(dataset[0][0])
        layers = []
        for out_dim in (512, 128, 32):
            layers.append(nn.Linear(input_dim, out_dim))
            layers.append(nn.ELU())
            input_dim = out_dim
        layers.append(nn.Linear(32, 1))
        if binary:
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        # Loss & optimizer
        self.criterion = (
            nn.BCELoss() if binary else nn.MSELoss()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def save(self, path: Path) -> None:
        """Override: include reserved sections info"""
        super().save(path)  # base handles majority
        # Optionally, save reserved sections list
        torch.save({'reserved': self.reserved_sections.tolist()}, path.with_suffix('.meta'))

    @classmethod
    def load(
        cls,
        path: Path,
        dataset: PixelDataset,
        device=None
    ) -> 'PixelClassifier':
        # Load base checkpoint
        base = super().load(path, device)
        # Extract parameters from run_name or meta
        # For now, assume same init args
        model = cls(
            dataset=dataset,
            batch_size=base.batch_size,
            learning_rate=base.lr,
            timestamp=False,
            binary=base.binary
        )
        model.load_state_dict(base.state_dict())
        model.epochs_trained = base.epochs_trained
        model.history = base.history
        # load reserved meta if exists
        meta_path = path.with_suffix('.meta')
        if meta_path.exists():
            meta = torch.load(meta_path)
            model.reserved_sections = torch.tensor(meta['reserved'])
        return model
