from pathlib import Path
import torch
from torch.utils.data import random_split
from typing import Type
from .autoencoder_base import AutoencoderBase

# registry of model-name → class
MODEL_REGISTRY: dict[str, Type[AutoencoderBase]] = {}


def register_model(name: str, cls: Type[AutoencoderBase]):
    """
    Add entry to MODEL_REGISTRY[name] = cls
    """
    MODEL_REGISTRY[name] = cls


def load_model(path: Path) -> AutoencoderBase:
    """
    Extract prefix = path.stem.split('_')[0], look up in registry,
    and call cls.load(path).
    """
    prefix = path.stem.split("_")[0]
    cls = MODEL_REGISTRY[prefix]
    return cls.load(path)


def split_dataset(dataset, train_frac: float = 0.8, seed: int = None):
    """
    Seed (if given), then random_split into train/val by fraction.
    Returns (train_set, val_set).
    """
    if seed is not None:
        torch.manual_seed(seed)
    total = len(dataset)
    n_train = int(total * train_frac)
    n_val = total - n_train
    return random_split(dataset, [n_train, n_val])
