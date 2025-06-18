from pathlib import Path
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

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
_, val_set = split_dataset(dataset, train_frac=0.8, seed=42)

# 2. Val DataLoader (batch through entire validation set)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=False)

# 3. Load your pretrained model
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

chk = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(chk['state_dict'])
model.eval()

# 4. Iterate over ALL validation batches, collect per-sample MSEs
all_mse = []
with torch.no_grad():
    for waves in val_loader:
        waves = waves.unsqueeze(1).to(DEVICE)       # [B,1,L]
        recon = model(waves).unsqueeze(1).cpu()     # [B,1,L] on CPU
        mse_batch = torch.mean((recon - waves.cpu())**2, dim=[1,2])  # [B]
        all_mse.append(mse_batch.numpy())

# concatenate into one big array of shape [num_val_samples]
mse_per_sample = np.concatenate(all_mse, axis=0)

# 5. Plot histogram of the full validation-set errors
plt.figure(figsize=(8,5))
plt.hist(mse_per_sample, bins=50, edgecolor='black', log=True)
plt.xlabel('Reconstruction MSE per sample')
plt.ylabel('Count (log scale)')
plt.title('Validation Reconstruction Error Distribution')
thresh = np.percentile(mse_per_sample, 95)
plt.axvline(thresh, color='red', linestyle='--', label=f'95th %ile = {thresh:.2e}')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Print the top 5 worst-reconstructed samples
topk = np.argsort(mse_per_sample)[-5:][::-1]
print("Top 5 highest-error samples (index in val set → MSE):")
for rank, idx in enumerate(topk, 1):
    print(f"{rank:>2}) Sample {idx:>4} → MSE = {mse_per_sample[idx]:.3e}")
