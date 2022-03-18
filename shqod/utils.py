"""Utility functions, varied."""

from collections.abc import Iterable

import numpy as np


def _get_iterable(x):
    """Utility function."""
    if isinstance(x, Iterable) and not isinstance(x, str):
        return x
    else:
        return (x,)


def sigmoid_ftn(x):
    return 1 / (1 + np.exp(-x))


def linear_extend(values, res):
    """
    Extend values by resolution by linear interpolation.

    (endpoint strictly inclusive)

    """
    values = np.asarray(values)
    if (ndim := values.ndim) not in (1, 2):
        raise ValueError

    if ndim == 1:
        values = values.reshape(-1, 1)  # column vector

    N = len(values)
    x = np.arange(1 + (N - 1) * res) / res

    out = np.vstack([np.interp(x, range(N), vals) for vals in values.T]).T
    if ndim == 1:
        out = out.flatten()

    return out
