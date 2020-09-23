"""Trajectories smoothing functionality."""

from .utils import linear_extend
from .dtypes import Trajec

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth(trajec: Trajec, spline_res: float = 3) -> Trajec:
    """Return smooth trajectory."""
    assert trajec.shape[1] == 2, 'trajec has the wrong shape'

    X = linear_extend(trajec[:, 0], spline_res)
    Y = linear_extend(trajec[:, 1], spline_res)
    gX = gaussian_filter(X, spline_res * 1.67)
    gY = gaussian_filter(Y, spline_res * 1.67)

    return np.vstack((gX, gY)).T
