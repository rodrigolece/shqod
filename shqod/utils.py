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


def gaussian_2d_filter(n, step_size):
    # Odd sized
    pre_output = abs(
        np.matmul(
            np.ones(2 * n + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
        )
    )
    pre_output += pre_output.transpose()
    pre_output = pre_output * pre_output
    pre_output *= -step_size / 2
    out = np.exp(pre_output) / np.sqrt(2 * np.pi)

    return out


def linear_extend(values, res):
    """
    Extend values by resolution by linear interpolation.

    (endpoint strictly inclusive)

    """
    n = values.size
    out = np.interp(np.arange(1 + (n - 1) * res) / res, np.arange(n), values)

    return out
