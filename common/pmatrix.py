import numpy as np
import cv2
import torch
from pathlib import Path
from .config import SECTIONS_DIR, colors

# Nota: esta versión es equivalente a la de Version 2 de Pepe. he hecho cambios para claridad

def Pmatrix(img, nx: int, ny: int,
            color_stone: np.ndarray = np.array(colors["Piedra"]),
            color_mortar: np.ndarray = np.array(colors["Mortero"]),
            interpolation: int = cv2.INTER_LINEAR):
    """
    Convert an RGB masonry image to a grayscale purity map and its binary mask.

    Parameters:
    - img: path to image file or numpy array (H×W×3)
    - nx, ny: desired output width and height
    - color_stone, color_mortar: RGB color vectors for stone and mortar
    - interpolation: cv2 interpolation flag

    Returns:
    - P: float array shape (ny, nx), grayscale purity between 0 (stone) and 1 (mortar)
    - P_flat: flattened P of length ny*nx
    - P_bin: binary mask array shape (ny, nx);
             0 for purity<0.5, 1 otherwise
    - P_bin_flat: flattened P_bin
    """
    # Load image if given a filepath
    if isinstance(img, (str, Path)):
        img = cv2.imread(str(img))

    # Resize to target dimensions
    resized = cv2.resize(img, (nx, ny), interpolation=interpolation)
    purity = np.zeros((ny, nx), dtype=float)
    color_dist = np.linalg.norm(color_stone - color_mortar)

    # Compute purity via projection onto stone–mortar line
    for j in range(ny):
        for i in range(nx):
            pixel = resized[j, i].astype(float)
            d_stone = np.linalg.norm(pixel - color_stone)
            d_mortar = np.linalg.norm(pixel - color_mortar)
            purity[j, i] = (d_stone**2 - d_mortar**2 + color_dist**2) / (2 * color_dist**2)

    P_flat = purity.flatten()
    binary = (purity >= 0.5).astype(float)
    binary_flat = binary.flatten()
    return purity, P_flat, binary, binary_flat


def tensor_pmatrix(folder: Path = Path(SECTIONS_DIR),
                   nx: int = None, ny: int = None,
                   color_stone: np.ndarray = np.array(colors["Piedra"]),
                   color_mortar: np.ndarray = np.array(colors["Mortero"]),
                   interpolation: int = cv2.INTER_LINEAR,
                   binary: bool = True,
                   skips: list = [] ) -> tuple:
    """
    Process all masonry section images in a folder into a torch tensor.

    Parameters:
    - folder: Path containing image files named by integer stems
    - nx, ny: target image dimensions; if None, infer from Pmatrix call
    - color_stone, color_mortar: RGB vectors
    - interpolation: cv2 flag
    - binary: if True, return binary masks; else grayscale
    - skips: list of integer stems to omit

    Returns:
    - filenames: list of Path objects processed
    - tensor: torch.FloatTensor of shape (n_images, ny, nx)
    """
    files = sorted([f for f in folder.glob('*') if f.stem.isdigit() and int(f.stem) not in skips], key=lambda x: int(x.stem))
    images = []
    for f in files:
        P, _, P_bin, _ = Pmatrix(f, nx, ny, color_stone, color_mortar, interpolation)
        images.append(P_bin if binary else P)

    data = np.stack(images)
    return files, torch.tensor(data, dtype=torch.float)