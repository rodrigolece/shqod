"""Utility functions for path features."""

from typing import Tuple, Dict

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree

from shqod.utils import sigmoid_ftn
from shqod.matrices import od_matrix  # breakup_by_flags
from .transform import path2mat, boxcounts


def _path_vel(path: np.ndarray) -> float:
    diff = path[1:] - path[:-1]

    return np.linalg.norm(diff, axis=1)


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
    return _path_vel(path).sum()


def avg_curvature(path: np.ndarray) -> float:
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
    # T = len(path)  # proxy for duration

    diff = path[1:] - path[:-1]
    vel = np.linalg.norm(diff, axis=1)
    length = vel.sum()

    # Remove stationary points
    idx = vel > 0
    diff = diff[idx]
    vel = vel[idx]

    diff = diff / vel[:, np.newaxis]
    curv = np.linalg.norm(diff[1:] - diff[:-1], axis=1)
    avg_curv = np.sum(curv) / length

    return avg_curv


def bdy_affinity(
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
    # T = len(path)  # proxy for duration
    length = path_length(path)

    ds, _ = KDTree(bdy_coords).query(path, k=1)  # second out arg is indices
    ds = ds.flatten()  # otherwise column vector

    ds_rescaled = 2 * scale * (ds - (rout + rin) / 2) / (rout - rin)
    sigmoid_vals = sigmoid_ftn(-ds_rescaled)
    affinity = sigmoid_vals.sum() / length

    return affinity


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
    pow2 = int(np.log2(min(width, length)))  # int rounds down

    sizes = 2 ** np.arange(1, pow2)
    counts = []

    for size in sizes:
        counts.append(boxcounts(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    dim = -coeffs[0]
    #  out (dim, sizes, counts) if return_boxcts else dim

    return dim


def frobenius_deviation(
    path: np.ndarray, grid_size: Tuple[int, int], normative_mat: sp.csr_matrix
) -> float:
    od_mat = od_matrix([path], grid_size)
    norm = np.linalg.norm((normative_mat - od_mat).toarray(), "fro")

    return norm


def supremum_deviation(
    path: np.ndarray, grid_size: Tuple[int, int], normative_mat: sp.csr_matrix
) -> float:
    od_mat = od_matrix([path], grid_size)
    norm = np.linalg.norm((normative_mat - od_mat).toarray(), np.inf)

    return norm


def sum_match(
    path: np.ndarray, grid_size: Tuple[int, int], normative_mat: sp.csr_matrix
) -> float:
    od_mat = od_matrix([path], grid_size)
    r, s = od_mat.nonzero()
    match = normative_mat[r, s].sum() / len(r)

    return -match  # minus sign to reverse order


def mobility_functional(
    path: np.ndarray,
    field: Dict[Tuple[int, int], np.ndarray],
) -> float:
    """
    Short summary.

    Parameters
    ----------
    path : np.ndarray
        The (x, y) path for which the functional is computed.
    normative_mat : csr_matrix
        The orgin-destination (OD) matrix to use as input.
    grid_width : int
        The width of the grid in the level.
    flags : np.ndarray, optional
        The (x, y) coordinates of the flags (with the right order).

    Returns
    -------
    float

    """
    T = len(path)  # proxy for duration
    out = 0.0

    diff = path[1:] - path[:-1]

    #  if isinstance(normative_mat, list):
    #      assert flags is not None, "error: provide the coodinates of the flags"
    #      field = [mobility_field(mat, grid_width) for mat in normative_mat]

    #      idx = breakup_by_flags(path, flags, R=3)

    #      for k, sub_arr in enumerate(np.split(path, idx[:-1])):
    #          for el in sub_arr:
    #              Fi = field[k].get(tuple(el), np.zeros(2))
    #              #  out += np.dot(Fi, el)
    #              # TODO: this still has typo, should be diff not el

    #  else:

    for k, el in enumerate(path[:-1]):
        Fi = field.get(tuple(el), np.zeros(2))
        out += np.dot(Fi, diff[k])

    return -out / T  # minus sign to reverse order


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
