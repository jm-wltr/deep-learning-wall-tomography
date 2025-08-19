from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
import matplotlib.pyplot as plt
import random

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

    n_train = int(len(remaining_subset) * train_frac)
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
            nn.BCELoss(reduction='sum') if binary else nn.MSELoss()
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
        dataset: "PixelDataset",
        device=None
    ) -> "PixelClassifier":
        # Load checkpoint dict directly (do NOT call super().load)
        checkpoint = torch.load(path, map_location=device or DEVICE)

        # Recreate classifier with the args your __init__ actually supports
        model = cls(
            dataset=dataset,
            batch_size=checkpoint.get("batch_size", 32),
            learning_rate=checkpoint.get("learning_rate", 1e-3),
            # If you want to keep these consistent with training, persist them in .meta and load below
            timestamp=False,
            binary=checkpoint.get("binary", True),
        ).to(device or DEVICE)

        # Restore weights & training state
        print(checkpoint["state_dict"])
        model.load_state_dict(checkpoint["state_dict"])
        model.epochs_trained = checkpoint.get("epochs_trained", 0)
        model.history = checkpoint.get("history", {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        })

        # Keep original run name (useful for logs)
        rn = checkpoint.get("run_name")
        if rn:
            model.run_name = rn

        # Load reserved/meta if present
        meta_path = path.with_suffix(".meta")
        if meta_path.exists():
            meta = torch.load(meta_path, map_location=device or DEVICE)
            reserved = meta.get("reserved")
            if reserved is not None:
                model.reserved_sections = torch.tensor(reserved, dtype=torch.long)
            # Optional: if you saved these, you can adapt loaders/splits here
            model.train_frac = meta.get("train_frac", getattr(model, "train_frac", 0.8))
            model.seed = meta.get("seed", getattr(model, "seed", None))

        model.eval()
        return model



    def plot_reconstructions(self):
        # plot 5 test (reserved) and 5 random training sections, with grayscale and binarized preds
        nX, nY = self.dataset.nX, self.dataset.nY
        DIM = nX * nY

        # determine sections
        test_secs = self.reserved_sections.tolist()
        all_secs = list(range(self.dataset.num_sections))
        train_secs = [s for s in all_secs if s not in test_secs]
        random.seed(self.epochs_trained)  # reproducibility per epoch
        train_plot = random.sample(train_secs, min(5, len(train_secs)))
        test_plot = test_secs[:5]

        # setup figure: 6 rows (actual, gray pred, binary pred for test/train) × 5 columns
        fig, axes = plt.subplots(6, 5, figsize=(15, 18))
        fig.subplots_adjust(top=0.92)
        fig.suptitle(f'Reconstructions @ epoch {self.epochs_trained}', fontsize=16, y=0.98)

        def plot_block(secs, row_offset):
            for col, sec in enumerate(secs):
                start, end = sec * DIM, (sec + 1) * DIM
                subset = Subset(self.dataset, list(range(start, end)))
                loader = DataLoader(subset, batch_size=DIM, shuffle=False)
                feats, labs = next(iter(loader))
                feats = feats.to(self.device)
                preds_gray = self(feats).cpu().view(nY, nX).detach()
                preds_bin = (preds_gray > 0.5).float()
                labs2 = labs.view(nY, nX)

                # actual
                ax_act = axes[row_offset, col]
                ax_act.imshow(labs2, cmap='gray')
                ax_act.axis('off')
                if col == 0:
                    ax_act.set_ylabel('Actual', fontsize=12, fontweight='bold')

                # grayscale prediction
                ax_gray = axes[row_offset + 1, col]
                ax_gray.imshow(preds_gray, cmap='gray')
                ax_gray.axis('off')
                if col == 0:
                    ax_gray.set_ylabel('Pred Gray', fontsize=12, fontweight='bold')

                # binary prediction
                ax_bin = axes[row_offset + 2, col]
                ax_bin.imshow(preds_bin, cmap='gray')
                ax_bin.axis('off')
                if col == 0:
                    ax_bin.set_ylabel('Pred Bin', fontsize=12, fontweight='bold')

        # test block in rows 0-2
        plot_block(test_plot, row_offset=0)
        axes[0, 0].text(-0.5, -0.3, 'Test Sections', transform=axes[0, 0].transAxes,
                       fontsize=14, fontweight='bold', va='top')
        # train block in rows 3-5
        plot_block(train_plot, row_offset=3)
        axes[3, 0].text(-0.5, -0.3, 'Train Sections', transform=axes[3, 0].transAxes,
                       fontsize=14, fontweight='bold', va='top')

        # log to TensorBoard
        self.writer.add_figure('Recon_Train_vs_Test', fig, self.epochs_trained)
        plt.close(fig)
