from pathlib import Path
import torch
import matplotlib.pyplot as plt
import sys

def find_project_root(marker_dir: str = "artifacts") -> Path:
    start = Path(__file__).resolve()
    for parent in (start, *start.parents):
        if (parent / marker_dir).is_dir():
            return parent
    return Path.cwd().resolve()

PROJECT_ROOT = find_project_root("artifacts")
sys.path.insert(0, str(PROJECT_ROOT)) 

from common.config import DEVICE
from models.autoencoder.dataset_autoencoder import DatasetAutoencoder
from models.autoencoder.utils import split_dataset
from models.autoencoder.architectures.flexible_autoencoder import ConvAutoencoder

# 1. Recreate dataset & split
dataset = DatasetAutoencoder(
    path="data/waveforms",
    reduction="resample",
    n=200,
    save=False,
    force_reload=False
)
train_set, val_set = split_dataset(dataset, train_frac=0.8, seed=42)

# 2. Val DataLoader of exactly 30 samples
val_loader = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=False)

# 3. Grab one batch of 30 waves
waves = next(iter(val_loader))           # shape: [30, length]
waves = waves.unsqueeze(1).to(DEVICE)    # add channel dim => [30,1,length]

# 4. Instantiate model and load checkpoint
ckpt_path = Path("artifacts/autoencoder/checkpoints") / "ConvAE_resample200_lat32_bn_2025-06-17_16-08-45.pt"
model = ConvAutoencoder(
    dataset=dataset,
    latent_dim=32,
    dropout=0.0,
    use_batchnorm=True,
    batch_size=50,
    learning_rate=1e-3,
    train_frac=0.8,
    seed=42,
    reduction="resample",
    reduction_n=200,
    timestamp=False
).to(DEVICE)

# load state dict
chk = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(chk['state_dict'])
model.eval()

# 5. Compute reconstructions
with torch.no_grad():
    recon = model(waves).unsqueeze(1).cpu()  # back to [30,1,length] on CPU

# 6. Plot originals vs reconstructions
fig, axes = plt.subplots(10, 5, figsize=(15, 12))
for idx, ax in enumerate(axes.flatten()):
    orig = waves.cpu()[idx].squeeze()
    rec  = recon[idx].squeeze()
    ax.plot(orig, label='Orig')
    ax.plot(rec,  label='Recon')
    ax.set_xticks([]); ax.set_yticks([])

# Add a single legend (optional)
axes[0,0].legend(loc='upper right')
plt.tight_layout()
plt.show()


