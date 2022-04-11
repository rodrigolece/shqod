from itertools import groupby

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def visiting_order(
    path: np.ndarray,
    flags: np.ndarray,
    R: int = 3,
    safe_mode: bool = False,
) -> np.ndarray:

    """Calculate the order in which the flags are visited.

    Calling this function with the smoothened paths will be inefficient.

    Parameters
    ----------
    path : np.ndarray
        The (x, y) path for which the visiting order is calculated.
    flags : np.ndarray
        The (x, y) coordinates of the flags (with the right order).
    R : float, optional
        The radius around which the path is considered to have circled
        the flags.
    safe_mode : bool, optional
        Test the fist and the last flag (default is False).

    Returns
    -------
    np.ndarray
        The order in which the flags are actually visited.

    """
    dmat = distance_matrix(path, flags)
    _, j = (dmat < R).nonzero()
    out = [x[0] for x in groupby(j)]

    if safe_mode and out[0] != 0:
        warnings.warn("unexpeted first flag")
    if safe_mode and out[-1] != len(flags) - 1:
        warnings.warn("unexpected last flag")

    return np.array(out)


def vo_correctness(vo_series, lvl, verbose=True):
    """
    Take an input series with the visiting orders and determine whether they are
    in the correct order for the level.

    Parameters
    ----------
    vo_series : pd.Series
        Series containing the visiting order in the form of arrays.
    lvl : int
        The level; used to determine what is the expected order.
    verbose : bool
        Whether to print the fraction of correct paths (default is True).

    pd.Series
        Boolean series indicating whether the order is correct or incorrect.

    """
    if not isinstance(vo_series, pd.Series):
        raise ValueError

    # TODO: fill in other levels.
    orders = {
        1: [0],
        2: [0],
        6: [0, 1, 2],
        8: [0, 1, 2],
        11: [1, 0, 1, 2],
    }

    if lvl not in orders.keys():
        raise NotImplementedError(f"order not known for level {lvl}")

    target_order = orders[lvl]

    # To be able to deal wiht NaNs we will with empty array
    out = vo_series.apply(
        lambda x: x if isinstance(x, np.ndarray) else np.array([])
    ).apply(lambda el: el.tolist() == target_order)

    if verbose:
        fraction = out.sum() / len(vo_series)
        print(f"Matching: {fraction:.3f}")

    return out
