"""Utility functions, varied."""

from collections.abc import Iterable
import re

import numpy as np
from scipy import stats


def _get_iterable(x):
    """Utility function."""
    out = x if isinstance(x, Iterable) and not isinstance(x, str) else (x,)

    return out


def parse_age(age_str):
    """Utility function."""
    match = re.search("^(\d*):(\d*)$", age_str)
    if match:
        low, high = match.groups()
        low = 0 if low == "" else int(low)
        high = 99 if high == "" else int(high)
    else:
        raise ValueError("invalid format for age")

    return low, high


def confidence_interval(auc, std, alpha=0.95):
    offset = (1 - alpha) / 2
    percentiles = np.array([offset, 1 - offset])

    out = stats.norm.ppf(percentiles, loc=auc, scale=std)
    out[out > 1] = 1

    return out


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
