"""Path smoothing functionality."""

from .utils import linear_extend
from .dtypes import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth(path: Path, spline_res: float = 3) -> Path:
    """Return smooth path."""
    assert path.shape[1] == 2, "path has the wrong shape"

    X = linear_extend(path[:, 0], spline_res)
    Y = linear_extend(path[:, 1], spline_res)
    gX = gaussian_filter(X, spline_res * 1.67)
    gY = gaussian_filter(Y, spline_res * 1.67)

    return np.vstack((gX, gY)).T
