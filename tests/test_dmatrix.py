# Simple pytest tests for common/dmatrix.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from common.dmatrix import ray_length_in_voxel, ray_distance_matrix
import matplotlib.pyplot as plt

def test_ray_length_in_voxel_full_pass():
    # Ray fully passes through unit cube: length = sqrt(3)
    p0 = np.array([0.,0.,0.])
    p1 = np.array([1.,1.,1.])
    voxel = np.array([[0,1],[0,1],[0,1]])
    length = ray_length_in_voxel(p0, p1, voxel)
    assert pytest.approx(np.linalg.norm(p1-p0), rel=1e-6) == length

def test_ray_length_known_diagonal():
    # Based on example from Datos.ipynb in the original project
    grid = np.array([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])
    P0 = np.array([0.75, 0.75, 1])
    P1 = np.array([0.5, 0.5, 0])
    expected = 0.424264
    result = ray_length_in_voxel(P0, P1, grid)
    print(f"\nExpected: {expected:.6f}\tResult: {result:.6f}")
    assert np.isclose(result, expected, atol=1e-4)


def test_ray_distance_matrix_non_intersect():
    # Ray outside cube yields zero matrix
    p0 = np.array([2.,2.,2.])
    p1 = np.array([3.,3.,3.])
    mat = ray_distance_matrix(p0, p1, nx=2, ny=2)
    assert mat.shape == (2,2)
    assert np.all(mat == 0)