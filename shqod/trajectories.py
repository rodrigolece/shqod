"""Utility functions for trajectories."""

from typing import Optional, Tuple
from .dtypes import Trajec, MatTrajec, BoxCounts

import numpy as np


def trajec2mat(trajec: Trajec, width: int, length: int) -> MatTrajec:
    """Convert a trajectory (x-y) to a matrix containing non-zero elements.

    Parameters
    ----------
    trajec : Trajec
        The trjectory to convert.
    width, length : int
        The dimensions of the level grid.

    Returns
    -------
    np.ndarray
        A matrix contaning the trajectory as non-zero elements.

    """
    out = np.zeros((width, length), dtype=int)

    for x, y in trajec:
        out[x, y] += 1

    return out


def boxcounts(Z: MatTrajec, box_size: int) -> int:
    """Calculate the number of non-empty boxes of given size.

    Parameters
    ----------
    Z : np.ndarray
        A matrix containing the trajectory as non-zero elements
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


def fractalD(
    trajec: Trajec, width: int, length: int, return_boxcts: bool = False
) -> Tuple[int, Optional[BoxCounts]]:
    """Calculate the Fractal dimension of a trajectory.

    Parameters
    ----------
    trajec : Trajec
        The trjectory to convert.
    width, length : int
        The dimensions of the level grid.
    return_boxcts : bool, optional
        Return the bin sizes and the number of counts (default is False).

    Returns
    -------
    float
        Thre fractal dimension.
    sizes, counts : (int, int), optional
        The size and number of non-empty boxes.

    """
    Z = trajec2mat(trajec, width, length)

    # Greatest power of 2 less than or equal to min size
    N = int(np.log2(min(width, length)))  # int rounds down

    sizes = 2 ** np.arange(1, N)  # +1 so that N is included
    counts = []

    for size in sizes:
        counts.append(boxcounts(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    D = -coeffs[0]

    return (D, sizes, counts) if return_boxcts else D


def trajectory_length(trajec: Trajec) -> float:
    """Calculate the total trrajectory length.

    Parameters
    ----------

    Returns
    -------

    """
    mat = np.array(list(trajec))

    return np.linalg.norm(mat[1:] - mat[:-1], axis=1).sum()
