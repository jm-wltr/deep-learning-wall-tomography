# Simple pytest tests for common/pmatrix.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import torch
from common.pmatrix import Pmatrix, tensor_pmatrix

def make_test_image(nx, ny, top_color, bottom_color):
    # Top half one color, bottom half another
    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[:ny//2] = top_color
    img[ny//2:] = bottom_color
    return img


def test_Pmatrix_binary_map():
    nx, ny = 4,4
    stone = np.array([0,0,0],np.uint8)
    mortar = np.array([255,255,255],np.uint8)
    img = make_test_image(nx, ny, stone, mortar)
    P,_,P_bin,_ = Pmatrix(img, nx, ny, stone, mortar, interpolation=cv2.INTER_NEAREST)
    # Expect top half zero, bottom half ones
    assert P_bin.sum() == (nx*(ny//2))


def test_tensor_pmatrix(tmp_path):
    # Write two small images and load
    dirp = tmp_path / 'sections'
    dirp.mkdir()
    stone = np.array([0,0,0],np.uint8)
    mortar = np.array([255,255,255],np.uint8)
    for i, col in enumerate([stone, mortar]):
        img = make_test_image(2,2,col,col)
        cv2.imwrite(str(dirp / f"{i}.png"), img)
    files, tensor = tensor_pmatrix(dirp, nx=2, ny=2, binary=True)
    assert isinstance(tensor, torch.FloatTensor)
    assert tensor.shape == (2,2,2)
