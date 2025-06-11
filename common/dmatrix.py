import warnings
import numpy as np
import pandas as pd
from .config import dims

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Column names for ray data files
COLUMNS = ["Xe","Ye","Ze","Xr","Yr","Zr","Velocity","Amplitude","Frequency"]

def ray_length_in_voxel(p0: np.ndarray, p1: np.ndarray, voxel: np.ndarray) -> float:
    """
    Compute how far a ray (segment from p0 to p1) travels inside a voxel.

    Parameters:
    - p0: Start point of ray (3D array)
    - p1: End point of ray (3D array)
    - voxel: 3x2 array defining the min and max bounds along x, y, z

    Returns:
    - Length of segment inside the voxel, or 0 if it doesn't intersect
    """
    total_len = np.linalg.norm(p1 - p0)
    t = np.zeros((3,2))
    # Parametric intersections with voxel planes
    t[:,0] = (voxel[:,0] - p0) / (p1 - p0)
    t[:,1] = (voxel[:,1] - p0) / (p1 - p0)

    t_enter = np.nanmax(np.min(t, axis=1))
    t_exit  = np.nanmin(np.max(t, axis=1))

    # Check intersection within segment
    if 0 <= t_exit and t_enter <= 1 and t_enter <= t_exit:
        return total_len * (min(t_exit,1) - max(t_enter,0))
    return 0.0


def voxel_iterator(xmin: float, xmax: float, nx: int,
                   ymin: float, ymax: float, ny: int,
                   zmin: float=0, zmax: float=0, nz: int=0,
                   pad: int=1, include_edges: bool=True):
    """
    Iterate over voxel definitions in a structured 2D or 3D grid.

    Parameters:
    - [x/y/z]min, [x/y/z]max: domain bounds
    - n[x/y/z]: number of divisions in each direction
    - pad: how many voxels to skip near edges if include_edges=False
    - include_edges: whether to include boundary voxels

    Yields:
    - 3x2 array of bounds for each voxel: [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    """
    xs = np.linspace(xmin, xmax, nx+1)
    ys = np.linspace(ymin, ymax, ny+1)
    zs = np.linspace(zmin, zmax, nz+1) if nz>0 else [0]
    start = 0 if include_edges else pad
    end_x = nx if include_edges else nx - pad
    end_y = ny if include_edges else ny - pad
    end_z = (nz if include_edges else nz - pad) or 1

    for i in range(start, end_x):
        for j in range(start, end_y):
            for k in range(start, end_z):
                yield np.array([
                    [xs[max(i-pad,0)], xs[min(i+1+pad,nx)]],
                    [ys[max(j-pad,0)], ys[min(j+1+pad,ny)]],
                    [zs[max(k-pad,0)], zs[min(k+1+pad,nz)]]
                ])


def ray_distance_matrix(p0: np.ndarray, p1: np.ndarray,
                        nx: int, ny: int, nz: int=0,
                        pad: int=1, include_edges: bool=True) -> np.ndarray:
    """
    Compute voxel-wise intersection distances for a single ray.

    Parameters:
    - p0, p1: emission and reception points (3D)
    - nx, ny, nz: number of voxels in each direction
    - pad, include_edges: voxel filtering settings (see voxel_iterator)

    Returns:
    - Array of shape (nx, ny [, nz]) with segment length inside each voxel
    """
    xmin,xmax = dims[0]
    ymin,ymax = dims[1]
    zmin,zmax = dims[2]

    voxels = voxel_iterator(xmin,xmax,nx, ymin,ymax,ny,
                             zmin,zmax,nz, pad, include_edges)
    lengths = [ray_length_in_voxel(p0, p1, vox) for vox in voxels]

    shape = (nx, ny) if nz==0 else (nx, ny, nz)
    if not include_edges:
        shape = tuple(s - 2*pad for s in shape)
    return np.array(lengths).reshape(shape)


def load_ray_tensor(filepath: str,
                    nx: int, ny: int, nz: int=0,
                    pad: int=1, include_edges: bool=True) -> tuple:
    """
    Load ray data and build a tensor with intersection distances.

    Parameters:
    - filepath: path to the ray data file (.txt)
    - nx, ny, nz: number of voxels along each axis
    - pad, include_edges: voxel masking options

    Returns:
    - D: ndarray of shape (nx, ny [, nz], num_rays) = distance matrix
    - params: DataFrame with ['Velocity', 'Amplitude', 'Frequency']
    - dims: bounding box array [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    """
    df = pd.read_csv(filepath, sep='\t', names=COLUMNS)
    params = df[['Velocity','Amplitude','Frequency']]

    matrices = []
    for _, row in df.iterrows():
        p0 = row[['Xe','Ye','Ze']].values
        p1 = row[['Xr','Yr','Zr']].values
        matrices.append(
            ray_distance_matrix(p0, p1, nx, ny, nz, pad, include_edges)
        )

    D = np.stack(matrices, axis=-1)
    return D, params, np.array(dims)