"""Utility functions for path transformations."""

import numpy as np
from scipy.ndimage import gaussian_filter

from shqod.utils import linear_extend, gaussian_2d_filter


def path2mat(path: np.ndarray, width: int, length: int) -> np.ndarray:
    """Convert a path (x-y) to a matrix containing non-zero elements.

    Parameters
    ----------
    path : np.ndarray
        The trjectory to convert.
    width, length : int
        The dimensions of the level grid.

    Returns
    -------
    np.ndarray
        A matrix contaning the path as non-zero elements.

    """
    out = np.zeros((width, length), dtype=int)

    for x, y in path:
        out[x, y] += 1

    return out


def boxcounts(Z: np.ndarray, box_size: int) -> int:
    """Calculate the number of non-empty boxes of given size.

    Parameters
    ----------
    Z : np.ndarray
        A matrix containing the path as non-zero elements
    box_size : int
        The size of the boxes to use.

    Returns
    -------
    int
        The number of non-empty boxes.
    """
    length, width = Z.shape

    idx_xboxes = np.arange(0, width, box_size)
    idx_yboxes = np.arange(1, length, box_size)

    # this reduces along axis 1 (columns; x axis)
    count_x = np.add.reduceat(Z, idx_xboxes, axis=1)
    # idem along axis 0 (rows; y axis)
    counts = np.add.reduceat(count_x, idx_yboxes, axis=0)

    # Count non-empty
    out = len(np.where(counts > 0)[0])

    return out


def smooth(
    path: np.ndarray,
    spline_res: float = 3,
    bandwidth: float = 1.67,
) -> np.ndarray:
    """Return smooth path."""

    assert path.shape[1] == 2, "path has the wrong shape"

    X = linear_extend(path[:, 0], spline_res)
    Y = linear_extend(path[:, 1], spline_res)
    gX = gaussian_filter(X, spline_res * bandwidth)
    gY = gaussian_filter(Y, spline_res * bandwidth)

    return np.vstack((gX, gY)).T
