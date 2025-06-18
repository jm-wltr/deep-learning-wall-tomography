import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import math

def find_project_root(marker_dir: str = "artifacts") -> Path:
    start = Path(__file__).resolve()
    for parent in (start, *start.parents):
        if (parent / marker_dir).is_dir():
            return parent
    return Path.cwd().resolve()

# 1) Locate project root
PROJECT_ROOT = find_project_root("artifacts")
sys.path.insert(0, str(PROJECT_ROOT))

# 2) Imports
from common.config import DEVICE
from models.autoencoder.dataset_autoencoder import DatasetAutoencoder
from models.autoencoder.architectures.flexible_autoencoder import ConvAutoencoder

# 3) Load full dataset
dataset = DatasetAutoencoder(
    path=PROJECT_ROOT / "data" / "waveforms",
    reduction="resample",
    n=200,
    save=False,
    force_reload=False
)

# 4) Load best model checkpoint
run_name = "ConvAE_resample200_lat32_bn_2025-06-17_13-49-26"
ckpt = torch.load(
    PROJECT_ROOT / "artifacts/autoencoder/checkpoints" / f"{run_name}.pt",
    map_location="cpu"
)

# 5) Recreate & load model
model = ConvAutoencoder(
    dataset=dataset, latent_dim=32, dropout=0.0, use_batchnorm=True,
    batch_size=32, learning_rate=1e-3, train_frac=0.8, seed=None,
    reduction="resample", reduction_n=200
)
state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
if isinstance(state, dict):
    model.load_state_dict(state)
elif hasattr(state, "state_dict"):
    model = state
model = model.to(DEVICE)
model.eval()

# 6) Gather all validation samples
val_loader = model.val_loader
val_samples = []
for batch in val_loader:
    # if dataset returns tuples, unpack first element
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    for sample in batch:
        # sample shape [1, L] or [L]
        arr = sample.cpu().numpy().squeeze()
        val_samples.append(arr)

# 7) Plot validation set in 5×5 grid
cols, rows = 5, 5
samples_per_page = cols * rows
total = len(val_samples)
pages = math.ceil(total / samples_per_page)

for page in range(pages):
    start = page * samples_per_page
    end = min(start + samples_per_page, total)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, idx in enumerate(range(start, end)):
        ax = axes_flat[i]
        y = val_samples[idx]
        x = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon = model(x).cpu().numpy().squeeze()

        ax.plot(y, linewidth=1)
        ax.plot(recon, linestyle="--", linewidth=1)
        ax.set_title(f"Val Sample {idx}", fontsize=8)
        ax.grid(True)

    # Hide any unused subplots
    for j in range(end - start, samples_per_page):
        axes_flat[j].axis("off")

    # Single legend
    handles = [
        plt.Line2D([0], [0], color='C0', lw=1, label='Original'),
        plt.Line2D([0], [0], color='C0', lw=1, linestyle='--', label='Reconstruction')
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    fig.suptitle(f"Validation Reconstructions Page {page+1}/{pages}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()

    if page < pages - 1:
        input("Press Enter to view next page...")
        
