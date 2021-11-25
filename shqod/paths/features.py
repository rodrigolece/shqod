"""Utility functions for path features."""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree

from shqod.utils import sigmoid_ftn
from paths.transform import path2mat, boxcounts


def path_length(path: np.ndarray) -> float:
    """
    Total length of a path.

    Parameters
    ----------
    path : np.ndarray
        The input path.

    Returns
    -------
    float

    """
    diff = path[1:] - path[:-1]

    return np.linalg.norm(diff, axis=1).sum()


def path_curvature(path: np.ndarray) -> float:
    """
    Average curvature of a path.

    Parameters
    ----------
    path : np.ndarray
        The input path.

    Returns
    -------
    float

    """
    T = len(path)  # proxy for duration

    diff = path[1:] - path[:-1]
    vel = np.linalg.norm(diff, axis=1)

    # Remove stationary points
    idx = vel > 0
    diff = diff[idx]
    vel = vel[idx]

    diff = diff / vel[:, np.newaxis]
    curv = np.linalg.norm(diff[1:] - diff[:-1], axis=1)
    avg_curv = np.sum(curv) / T

    return avg_curv


def path_bdy(
    path: np.ndarray,
    bdy_coords: np.ndarray,
    rin: float = 1.5,
    rout: float = 5,
    scale: float = 4,
) -> float:
    """
    Boundary affinity of a path.

    Parameters
    ----------
    path : np.ndarray
        The input path.
    bdy_coords np.ndarray
        2D array with the coordinates of the boundary.
    rin, rout : float
        The cut-off values for the sigmoid function.
    scale : float
        Re-scaling factor.

    Returns
    -------
    float

    """
    T = len(path)  # proxy for duration

    ds, _ = KDTree(bdy_coords).query(path, k=1)  # second out arg is indices
    ds = ds.flatten()  # otherwise column vector

    ds_rescaled = 2 * scale * (ds - (rout + rin) / 2) / (rout - rin)
    sigmoid_vals = sigmoid_ftn(-ds_rescaled) / T

    return sigmoid_vals.sum()


def fractal_dim(path: np.ndarray, width: int, length: int) -> float:
    """
    The Fractal dimension of a path.

    Parameters
    ----------
    path : np.ndarray
        The input path.
    width, length : int
        The dimensions of the level grid.

    Returns
    -------
    float

    """
    Z = path2mat(path, width, length)

    # Greatest power of 2 less than or equal to min size
    N = int(np.log2(min(width, length)))  # int rounds down

    sizes = 2 ** np.arange(1, N)  # +1 so that N is included
    counts = []

    for size in sizes:
        counts.append(boxcounts(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    dim = -coeffs[0]
    #  out (dim, sizes, counts) if return_boxcts else dim

    return dim


def path_fro(path: np.ndarray, normative_mat: sp.csr_matrix) -> float:
    pass


def path_sup(path: np.ndarray, normative_mat: sp.csr_matrix) -> float:
    pass


def path_match(path: np.ndarray, normative_mat: sp.csr_matrix) -> float:
    pass


#  def path_integrate(velocity):
#      """
#      No velocity consideration, assume uniform time.

#      Parameters
#      ----------
#      velocity : np.ndarray
#          Each row is the velocity at each time.

#      Returns
#      -------
#      np.ndarray
#          The array of cumulative sums.

#      """
#      zero = np.zeros((1, velocity.shape[1]))
#      cumsum = np.cumsum(velocity, axis=0)

#      return np.vstack((zero, cumsum))
