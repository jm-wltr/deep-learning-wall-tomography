"""
Waveform I/O and reduction utilities for autoencoder dataset.

This module provides functions to read raw SLA waveform simulations
from the "numerical analyses" folder and apply various data reduction
methods (resampling, windowed mean, windowed max deviation).

Functions:
  - reduce_waveform(waveform, method, size)
  - load_emission_waveforms(emission_file, method, size)
  - load_sample_waveforms(sample_dir, method, size)
  - load_all_waveforms(root_dir, method, size)
  - load_waveforms_by_sample(root_dir, method, size, skips) 
"""
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.signal import resample

# Compute device
from common.config import DEVICE


def reduce_waveform(onda: torch.Tensor, modo: str = "max", n: int = 0) -> torch.Tensor:
    """
    Reduce the number of points in a waveform.

    Parameters:
        onda: 1D tensor of waveform values
        modo: 'resample', 'mean', 'max', or '' (no reduction except centering)
        n: reduction parameter (target length or window size)

    Returns:
        Reduced waveform tensor
    """
    # Center the waveform
    onda = onda - onda.mean()
    if modo == "resample":
        # FFT-based resampling to length n
        onda = torch.tensor(resample(onda.cpu().numpy(), n),
                            dtype=torch.float32,
                            device=DEVICE)
    elif modo == "mean":
        # Non-overlapping windowed mean
        onda = onda.unfold(0, n, n).mean(dim=1)
    elif modo == "max":
        # Windowed max deviation from zero
        windows = onda.unfold(0, n, n)
        idx = windows.abs().argmax(dim=1)
        onda = windows[torch.arange(len(windows)), idx]
    # else: no additional reduction, keep centering
    return onda


def load_emission_waveforms(archivo: Path, modo: str = "", n: int = 0) -> torch.Tensor:
    """
    Read a single emission file (PL*.txt) and reduce each of its 11 waveforms.

    Parameters:
        archivo: Path to PL*.txt file
        modo: reduction mode passed to reducir_onda
        n: reduction parameter

    Returns:
        Tensor of shape (11, reduced_length)
    """
    # Read time + 11 sensor columns; skip header lines starting with '%'
    df = pd.read_csv(archivo, comment='%', sep=r'\s+')
    # Some files may have extra columns; take the last 11 columns
    data = df.iloc[:, -11:].values.T  # shape (11, original_length)
    ondas = torch.tensor(data, dtype=torch.float32, device=DEVICE)
    # Apply reduction to each waveform
    reducidas = [reduce_waveform(onda, modo, n) for onda in ondas]
    return torch.stack(reducidas, dim=0)


def load_sample_waveforms(carpeta: Path, modo: str = "", n: int = 0) -> torch.Tensor:
    """
    Load and concatenate all emissions for one sample (probeta).

    Parameters:
        carpeta: Path to directory containing PL*.txt files for one sample
        modo, n: passed to leer_emision

    Returns:
        Tensor of shape (num_rays, reduced_length)
    """
    archivos = sorted(carpeta.glob('PL*.txt'))
    emisiones = [load_emission_waveforms(f, modo, n) for f in archivos]
    # Stack along ray axis: each file yields 11, total rays = 6*11=66
    return torch.cat(emisiones, dim=0)


def load_all_waveforms(carpeta: Path, reduccion: str = "", n: int = 0) -> torch.Tensor:
    """
    Load all waveforms across all samples, ignoring sample boundaries.

    Parameters:
        carpeta: Path to 'numerical analyses' root dir
        reduccion, n: passed to ondas_probeta

    Returns:
        Tensor of shape (total_rays, reduced_length)
    """
    carpetas = sorted([d for d in carpeta.iterdir() if d.is_dir()])
    datos = [load_sample_waveforms(d, reduccion, n) for d in carpetas]
    return torch.cat(datos, dim=0)


def load_waveforms_by_sample(carpeta: Path, reduccion: str = "", n: int = 0, skips: list = []) -> torch.Tensor:
    """
    Load waveforms organized by sample, skipping specified bad samples.

    Parameters:
        carpeta: Path to 'numerical analyses'
        reduccion, n: passed to ondas_probeta
        skips: list of sample directory names (as ints) to omit

    Returns:
        Tensor of shape (num_samples, num_rays, reduced_length)
    """
    carpetas = sorted([d for d in carpeta.iterdir()
                       if d.is_dir() and int(d.name) not in skips],
                      key=lambda d: int(d.name))
    datos = [load_sample_waveforms(d, reduccion, n) for d in carpetas]
    return torch.stack(datos, dim=0)