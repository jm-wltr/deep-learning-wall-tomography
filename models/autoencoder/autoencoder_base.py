from common.config import DEVICE, BASE_DIR
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor


class AutoencoderBase(nn.Module):
    """
    Base class for training and logging any autoencoder model.

    Handles:
      - Device configuration via common.config.DEVICE.
      - Unique run naming with optional timestamp.
      - TensorBoard logging of description, training/validation loss curves, and reconstructions.
      - Training loop with both per-epoch and optional per-batch verbosity.
      - Utilities for evaluation and plotting.
    """

    def __init__(
        self,
        model_name: str = "Autoencoder",
        description: str = "",
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        logs_root: Path = Path(BASE_DIR) / 'artifacts' / 'autoencoder',
        timestamp: bool = True,
    ):
        super().__init__()

        # Save hyperparameters
        self.description = description
        self.batch_size = batch_size
        self.lr = learning_rate

        # Device and naming
        self.device = DEVICE
        self.to(self.device)
        if timestamp:
            ts = time.strftime('%Y-%m-%d_%H-%M-%S')
            self.run_name = f"{model_name}_{ts}"
        else:
            self.run_name = model_name

        # Logging setup
        self.log_dir = logs_root / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.writer.add_text("Description", description)
        self.writer.add_custom_scalars_multilinechart(
            ['Loss/Train', 'Loss/Validation'],
            category='Metrics',
            title='Train vs Validation Loss'
        )

        # Training history
        self.epochs_trained = 0
        self.history = {"train_loss": [], "val_loss": []}

    def train_model(
        self,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs: int,
        log_mode: str = "epoch",
    ) -> None:
        total_batches = len(train_loader)

        # Initial evaluation before any weight updates
        if self.epochs_trained == 0:
            train_loss = self.evaluate(train_loader, criterion)
            val_loss = self.evaluate(val_loader, criterion)
            self._log_metrics(train_loss, val_loss)

        # Main training loop
        for epoch in range(1, num_epochs + 1):
            current_epoch = self.epochs_trained + 1
            self.train()
            running_loss = 0.0
            count = 0
            start_time = time.perf_counter()

            if log_mode == "batch":
                print(f"\nEpoch {current_epoch}/{self.epochs_trained + num_epochs}")

            for batch_idx, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                batch = batch.to(self.device)
                outputs = self(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch.size(0)
                count += batch.size(0)

                if log_mode == "batch":
                    elapsed = time.perf_counter() - start_time
                    avg = running_loss / count
                    print(f"\r Batch {batch_idx}/{total_batches} "
                          f"loss={avg:.4f}, time={elapsed:.1f}s", end="")

            # Epoch complete
            self.epochs_trained += 1
            avg_train = running_loss / count

            # Validation step
            self.eval()
            val_loss = self.evaluate(val_loader, criterion)

            if log_mode == "epoch":
                print(f"Epoch {self.epochs_trained}: train_loss={avg_train:.4f}, val_loss={val_loss:.4f}")

            # Record metrics and optional reconstructions
            self._log_metrics(avg_train, val_loss)
            if self.epochs_trained % 10 == 0 and hasattr(self, 'log_reconstruction'):
                fig = self.log_reconstruction(val_loader, criterion)
                self.writer.add_figure(f"Reconstruction", fig, self.epochs_trained)

    def evaluate(self, loader, criterion) -> float:
        total_loss = 0.0
        total_count = 0
        self.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs = self(batch)
                loss = criterion(outputs, batch)
                total_loss += loss.item() * batch.size(0)
                total_count += batch.size(0)
        return total_loss / total_count

    def _log_metrics(self, train_loss: float, val_loss: float) -> None:
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.writer.add_scalar('Loss/Train', train_loss, self.epochs_trained)
        self.writer.add_scalar('Loss/Validation', val_loss, self.epochs_trained)

    def plot_history(self) -> None:
        import matplotlib.pyplot as plt
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save(self, path: Path) -> None:
        checkpoint = {
            'state_dict': self.state_dict(),
            'epochs_trained': self.epochs_trained,
            'history': self.history,
            'run_name': self.run_name,
            'description': self.description,
            'batch_size': self.batch_size,
            'learning_rate': self.lr,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, device=None) -> 'AutoencoderBase':
        checkpoint = torch.load(path, map_location=device or DEVICE)
        model = cls(
            model_name=checkpoint.get('run_name', 'Autoencoder'),
            description=checkpoint.get('description', ''),
            batch_size=checkpoint.get('batch_size', 32),
            learning_rate=checkpoint.get('learning_rate', 1e-3),
            timestamp=False
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.epochs_trained = checkpoint['epochs_trained']
        model.history = checkpoint['history']
        return model

    def forward(self, x: Tensor) -> Tensor:
        x = self._ensure_batch_and_channel(x)
        return self.decoder(self.encoder(x.to(self.device))).squeeze(1)

    def encode(self, x: Tensor) -> Tensor:
        x = self._ensure_batch_and_channel(x)
        return self.encoder(x.to(self.device))

    @staticmethod
    def _ensure_batch_and_channel(x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x
