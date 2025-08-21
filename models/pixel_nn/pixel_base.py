from common.config import DEVICE, BASE_DIR
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor


class PixelBase(nn.Module):
    """
    Base class for training and logging pixel-wise classification/regression models.

    Handles:
      - Device configuration via common.config.DEVICE.
      - Unique run naming with optional timestamp.
      - TensorBoard logging for loss and accuracy.
      - Training loop with per-epoch or per-batch verbosity.
      - Evaluation utilities.
    """
    def __init__(
        self,
        model_name: str = "PixelModel",
        description: str = "",
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        logs_root: Path = Path(BASE_DIR) / 'artifacts' / 'pixel_nn',
        timestamp: bool = True,
        binary: bool = True,
    ):
        super().__init__()
        # Hyperparameters
        self.description = description
        self.batch_size = batch_size
        self.lr = learning_rate
        self.binary = binary  # classification vs. regression

        # Device setup
        self.device = DEVICE
        self.to(self.device)

        # Run naming
        if timestamp:
            ts = time.strftime('%Y-%m-%d_%H-%M-%S')
            self.run_name = f"{model_name}_{ts}"
        else:
            self.run_name = model_name

        # Logging directory
        self.log_dir = logs_root / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.writer.add_text("Description", description)
        # Configure custom scalars chart: loss and accuracy
        self.writer.add_custom_scalars_multilinechart(
            ['Loss/Train', 'Loss/Val', 'Acc/Train', 'Acc/Val'],
            category='Metrics',
            title='Loss & Accuracy'
        )

        # Training history
        self.epochs_trained = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

    def _binary_acc_from_outputs(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        Compute binary accuracy exactly like plot_reconstructions:
        - Threshold the RAW model outputs at `threshold` (no sigmoid).
        - Threshold the labels the same way.
        Works for both binary-classification (Sigmoid+BCELoss) and regression (MSE) models.

        Returns:
            float: mean accuracy over all valid elements.
        """
        with torch.no_grad():
            # Move labels to same device and align shape if it's just a reshape away
            if labels.device != outputs.device:
                labels = labels.to(outputs.device)
            if labels.shape != outputs.shape and labels.numel() == outputs.numel():
                labels = labels.reshape_as(outputs)

            # Ensure floating tensors
            outputs = outputs.float()
            labels  = labels.float()

            # Validity mask to ignore NaN/Inf if any
            valid = torch.isfinite(outputs) & torch.isfinite(labels)
            if not torch.any(valid):
                return float("nan")

            # Binarize EXACTLY as in plot_reconstructions (no sigmoid)
            preds_bin  = (outputs > threshold)
            labels_bin = (labels  > threshold)

            acc = (preds_bin[valid] == labels_bin[valid]).float().mean().item()
            return acc

    def train_model(
        self,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs: int,
        log_mode: str = 'epoch'
    ) -> None:
        total_batches = len(train_loader)

        # Initial evaluation before training
        if self.epochs_trained == 0:
            tr_loss, tr_acc = self.evaluate(train_loader, criterion)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            self._log_metrics(tr_loss, tr_acc, val_loss, val_acc)

        # Main training loop
        for _ in range(num_epochs):
            self.train()
            running_loss, running_corrects, count = 0.0, 0, 0
            start_time = time.perf_counter()

            for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Metrics
                running_loss += loss.item() * inputs.size(0)
                count += inputs.size(0)
                # Always compute binary accuracy by thresholding at 0.5
                epoch_acc_part = self._binary_acc_from_outputs(outputs.detach(), labels)
                running_corrects += epoch_acc_part * inputs.size(0)

                # Optional batch logging
                if log_mode == 'batch' and batch_idx % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    avg_loss = running_loss / count
                    avg_acc = running_corrects / count
                    print(f"Batch {batch_idx}/{total_batches} | "
                          f"loss={avg_loss:.4f}" +
                          (f", acc={avg_acc:.4f}" if self.binary else "") +
                          f", time={elapsed:.1f}s")

            # Epoch complete
            self.epochs_trained += 1
            epoch_loss = running_loss / count
            epoch_acc = running_corrects / count

            # Validation
            self.eval()
            val_loss, val_acc = self.evaluate(val_loader, criterion)

            # Epoch logging
            if log_mode == 'epoch':
                print(f"Epoch {self.epochs_trained} | "
                    f"train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"train_acc={epoch_acc:.4f}, val_acc={val_acc:.4f}")


            # Record in TensorBoard
            self._log_metrics(epoch_loss, epoch_acc, val_loss, val_acc)
            if self.epochs_trained % 5 == 0:
                self.plot_reconstructions()

    def evaluate(self, loader, criterion) -> tuple:
        total_loss, total_acc_sum, count = 0.0, 0.0, 0
        self.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                count += inputs.size(0)
                acc = self._binary_acc_from_outputs(outputs, labels)
                total_acc_sum += acc * inputs.size(0)
        avg_loss = total_loss / count
        avg_acc = total_acc_sum / count
        return avg_loss, avg_acc

    def _log_metrics(
        self,
        train_loss: float, train_acc: float,
        val_loss: float, val_acc: float
    ) -> None:
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        step = self.epochs_trained
        self.writer.add_scalar('Loss/Train', train_loss, step)
        self.writer.add_scalar('Loss/Val', val_loss, step)
        self.writer.add_scalar('Acc/Train', train_acc, step)
        self.writer.add_scalar('Acc/Val', val_acc, step)

    def plot_history(self) -> None:
        import matplotlib.pyplot as plt
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        if self.binary:
            plt.plot(self.history['train_acc'], label='Train Acc')
            plt.plot(self.history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
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
            'binary': self.binary
        }
        torch.save(checkpoint, path)
        
    def plot_reconstructions(self):
        """
        Placeholder: subclass should implement to plot two reserved sections.
        """
        raise NotImplementedError("Subclasses must implement plot_reconstructions()")

    @classmethod
    def load(
        cls,
        path: Path,
        device=None
    ) -> 'PixelBase':
        checkpoint = torch.load(path, map_location=device or DEVICE)
        model = cls(
            model_name=checkpoint.get('run_name','PixelModel'),
            description=checkpoint.get('description',''),
            batch_size=checkpoint.get('batch_size',32),
            learning_rate=checkpoint.get('learning_rate',1e-3),
            timestamp=False,
            binary=checkpoint.get('binary',True)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.epochs_trained = checkpoint['epochs_trained']
        model.history = checkpoint['history']
        return model