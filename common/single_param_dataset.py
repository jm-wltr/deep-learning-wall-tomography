import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, random_split

from .config import RAYS_DIR, SECTIONS_DIR, dims, resolution, DEVICE, colors
from .dmatrix import load_ray_tensor
from .pmatrix import tensor_pmatrix

# Número de rayos en los datos
N_RAYOS = dims[-1][-1] if len(dims) == 3 else 66

# Columnas esperadas en los archivos de rayos
COLUMNAS = [
    "Xe", "Ye", "Ze",
    "Xr", "Yr", "Zr",
    "Velocidad", "Amplitud",
    "Frec. Central"
]

# Parámetros de interpolación y geometría
ADY = 1
ESQUINAS = True


def Dmatrix(filename, parametro="Velocidad"):
    """
    Devuelve la matriz D (nx, ny, N_RAYOS) multiplicada por el parámetro.
    Utiliza load_ray_tensor para cargar distancias y parámetros.
    """
    filepath = os.path.join(RAYS_DIR, filename)
    probeta, params, _ = load_ray_tensor(filepath, *dims, ADY, ESQUINAS)
    if parametro == "Posicion":
        return probeta
    if parametro == "Random":
        return np.random.random(probeta.shape)
    if parametro == "RandPos":
        return probeta * np.random.random(probeta.shape)
    # caso normal: multiplicar por vector de parámetros
    return probeta * params[parametro].values


def apilarDmatrix(parametro="Velocidad"):
    """
    Construye tensor (n_samples, nx, ny, N_RAYOS) apilando Dmatrix para todos los archivos.
    Recalcula load_ray_tensor por cada muestra.
    """
    archivos = sorted(os.listdir(RAYS_DIR))
    mats = [Dmatrix(f, parametro) for f in archivos]
    return torch.tensor(np.stack(mats), device=DEVICE)


def generarDatos(parametro="Velocidad", m=2):
    """
    Genera datos eficientemente reutilizando un solo load_ray_tensor.
    Para random o Posicion/RandPos usa tensores aleatorios.
    """
    if parametro == "Random":
        nx, ny, _ = apilarDmatrix(parametro).shape[1:]
        return torch.rand(m, nx, ny, N_RAYOS, device=DEVICE)

    # cargar una sola probeta para distancias
    archivos = sorted(os.listdir(RAYS_DIR))
    probeta, _, _ = load_ray_tensor(os.path.join(RAYS_DIR, archivos[0]), *dims, ADY, ESQUINAS)
    probeta = torch.tensor(probeta, device=DEVICE)
    if parametro == "Posicion":
        return probeta.unsqueeze(0)
    if parametro == "RandPos":
        return torch.rand(m, *probeta.shape, device=DEVICE) * probeta

    # parámetros reales desde CSV
    params_list = []
    for archivo in archivos:
        df = pd.read_csv(os.path.join(RAYS_DIR, archivo), sep="\t", names=COLUMNAS)
        vec = torch.tensor(df[parametro].values, device=DEVICE)
        params_list.append(vec)
    params = torch.stack(params_list)
    # multiplicar distancia por cada parámetro
    data = probeta[None, ...] * params[:, None, None, :]
    return data


def generarEtiquetas(binarizada=True, skips=None):
    """
    Devuelve tensor (n_samples, nx, ny) de máscaras binarias o grises.
    Usa tensor_pmatrix sobre SECTIONS_DIR.
    """
    skips = skips or []
    # tensor_pmatrix devuelve (files, tensor)
    _, tensor = tensor_pmatrix(
        folder=SECTIONS_DIR,
        nx=int((dims[0][1]-dims[0][0])/resolution),
        ny=int((dims[1][1]-dims[1][0])/resolution),
        color_stone=np.array(colors["Piedra"]),
        color_mortar=np.array(colors["Mortero"]),
        interpolation=cv2.INTER_LINEAR,
        binary=binarizada,
        skips=skips
    )
    return tensor.to(device=DEVICE)


class DatasetConUnParametro(Dataset):  # noqa
    """
    Dataset que envuelve datos D y etiquetas P para un solo parámetro.
    Cada muestra es un píxel con sus entradas de rayos y etiqueta.
    """
    def __init__(self, parametro="Velocidad", binarizada=True):
        # generar tensores
        datos = generarDatos(parametro)
        etiq = generarEtiquetas(binarizada=binarizada)

        # extraer shapes
        self.N, nx, ny, nr = datos.shape
        # concatenar en filas de píxeles
        X = datos.reshape(self.N * nx * ny, nr)
        y = etiq.reshape(self.N * nx * ny)

        self.X = X.float()
        self.y = y.float().unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def generarParticion(self, reserva=2, train_ratio=0.8, seed=None):
        """
        Reserva probetas completas y crea train/validation splits.

        Returns: train_ds, val_ds, list_of_reserved_indices
        """
        if seed is not None:
            torch.manual_seed(seed)

        # seleccionar índices de probetas completas a reservar
        reservados_ids = torch.randperm(self.N)[:reserva].tolist()
        pixels_per = (self.X.shape[0] // self.N)
        reserved_idx = []
        for pid in reservados_ids:
            start = pid * pixels_per
            reserved_idx.extend(range(start, start + pixels_per))

        all_idx = list(range(self.X.shape[0]))
        normal_idx = list(set(all_idx) - set(reserved_idx))

        # crear subsets
        reserve_ds = Subset(self, reserved_idx)
        normal_ds = Subset(self, normal_idx)

        # dividir normal en train/val
        n_train = int(len(normal_ds) * train_ratio)
        train_ds, val_ds = random_split(normal_ds, [n_train, len(normal_ds) - n_train])
        # añadir reserva a val
        val_ds = ConcatDataset([val_ds, reserve_ds])

        torch.manual_seed(int(time.time()))
        return train_ds, val_ds, reservados_ids
