"""
Utilities for SHQ path analysis
"""

import sys
import math
from itertools import groupby

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree


def visiting_order(
    path: np.ndarray, flags: np.ndarray, R: int = 3, safe_mode: bool = False
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
        6: [0, 1, 2],
        8: [0, 1, 2],
        11: [1, 0, 1, 2],
    }

    if lvl not in orders.keys():
        raise NotImplementedError(f"order not known for level {lvl}")

    target_order = orders[lvl]

    out = vo_series.apply(lambda el: el.tolist() == target_order)

    if verbose:
        fraction = out.sum() / len(vo_series)
        print(f"Matching: {fraction:.3f}")

    return out


def path_velocity(path):
    """
    No velocity consideration, assume uniform time

    Parameters
    ----------
    path : np.ndarray
        The (x, y) path for which the visiting order is calculated.

    Returns
    -------
    np.ndarray
        The array of adjacent row-wise differences.

    """
    return path[1:] - path[:-1]


def path_integrate(velocity):
    """
    No velocity consideration, assume uniform time

    Parameters
    ----------
    velocity : np.ndarray
        Each row is the velocity at each time.

    Returns
    -------
    np.ndarray
        The array of cumulative sums.

    """
    return np.vstack((np.zeros((1, velocity.shape[1])), np.cumsum(velocity, axis=0)))


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
    output = np.exp(pre_output) / np.sqrt(2 * math.pi)
    return output


def filter_pointcloud(sample, labels_in, return_detailed=False):
    """
    Filter a pointcloud by integer label
    Assume that sample is a (k x n) list where there are n dimensions, and depth is a shape (k,) ndarray encoding labels
    Return a list of (m x n) arrays where m is the number of samples that has each label type.
    """
    labels = list(labels_in)
    num_samples = sample.shape[0]
    assert (
        len(labels) == num_samples
    ), "the number of samples must be the same as the number of depth data"
    unique_labels = np.unique(labels)
    unique_labels_hor_ext = np.matmul(
        unique_labels.reshape(-1, 1), np.ones((1, num_samples))
    )
    all_labels_vert_ext = np.matmul(
        np.ones((len(unique_labels), 1)), np.array(labels).reshape(1, -1)
    )
    sample_binary_labels = np.equal(
        unique_labels_hor_ext, all_labels_vert_ext
    )  # each column is a binary one-hot array indicating its label
    vanilla_indices = np.array(range(len(labels)))

    if not return_detailed:
        return [sample[sample_binary_labels[i]] for i in range(len(unique_labels))]
    else:
        return {
            "points": [
                sample[sample_binary_labels[i]] for i in range(len(unique_labels))
            ],
            "indices": [
                vanilla_indices[sample_binary_labels[i]]
                for i in range(len(unique_labels))
            ],
        }


def print_progress_bar(
    tick, max_tick, message="", percent_decimal=2, ETA_ESTIMATOR=None
):
    proportion = tick / max_tick
    proportion_prev = (tick - 1) / max_tick
    tick_d = math.floor((proportion) * 100 * (10 ** percent_decimal)) / (
        10 ** percent_decimal
    )
    tick_d_prev = math.floor((proportion_prev) * 100 * (10 ** percent_decimal)) / (
        10 ** percent_decimal
    )
    if tick_d > tick_d_prev:
        tick_d_str = str(tick_d)
        tick_d_str = ("0" * (3 - (tick_d_str.find(".")))) + tick_d_str
        tick_d_str = tick_d_str + ("0" * (3 - (len(tick_d_str) - tick_d_str.find("."))))
        if ETA_ESTIMATOR != None:
            sys.stdout.write("\r" + message + " " + tick_d_str + "%")
            sys.stdout.write(
                ("|" + "█" * int(tick_d / 5))
                + (" " * (20 - int(tick_d / 5)))
                + "| "
                + "ETA ≈ "
                + str(
                    datetime.timedelta(seconds=int((max_tick - tick) * ETA_ESTIMATOR))
                )
            )
        else:
            sys.stdout.write("\r" + message + tick_d_str + "%")
            sys.stdout.write(
                ("|" + "█" * int(tick_d / 5)) + (" " * (20 - int(tick_d / 5))) + "|"
            )
    if max_tick == tick:
        print("")


def path_curvature(path):
    T = len(path)  # proxy for duration

    tan = path_velocity(path)
    vel = np.linalg.norm(tan, axis=1)
    idx = vel > 0

    # Remove stationary points
    tan = tan[idx]
    vel = vel[idx]

    tan = tan / vel[:, np.newaxis]
    curv = np.linalg.norm(tan[1:] - tan[:-1], axis=1) / T
    total_curv = np.sum(curv)

    return total_curv


def path_length(path):
    vel = path_velocity(path)
    return np.sum(np.linalg.norm(vel, axis=1))


def sigmoid_ftn(x):
    return 1 / (1 + np.exp(-x))


def path_bdy(path, bdy_coords, rin=1.5, rout=5, scale=4):
    T = len(path)  # proxy for duration

    ds, _ = KDTree(bdy_coords).query(path, k=1)  # second out arg is indices
    ds = ds.flatten()  # otherwise column vector

    ds_rescaled = 2 * scale * (ds - (rout + rin) / 2) / (rout - rin)
    sigmoid_vals = sigmoid_ftn(-ds_rescaled) / T

    return np.sum(sigmoid_vals)


def linear_extend(values, res):
    """
    Extend values by resolution by linear interpolation (endpoint strictly inclusive)
    """
    n = values.size
    return np.interp(np.arange(1 + (n - 1) * res) / res, np.arange(n), values)


#  def clus_compare(clusA, clusB):
#      # Comparison accuracy of cluster labels
#      num_A = clusA.size
#      num_B = clusB.size
#      num_label_types_A = len(set(clusA))
#      num_label_types_B = len(set(clusB))
#      assert num_A == num_B
#      assert num_label_types_A == num_label_types_B

#      if num_label_types_A <= 2:
#          return max(np.sum(clusA == clusB), np.sum(clusA == 1-clusB)) / num_A
#      else:
#          raise RuntimeError("Not implemented Yet")

#  def labels_to_clus(labels):
#      # Changes labeled 1d array into the respective clusters.
#      # e.g. array([1,2,1,1,2]) into [array([0,2,3]), array([1,4])]
#      # :param labels: 1d array of labels
#      # :return: List of 1d arrays by labels
#      label_types = np.array(list(set(labels))) # Removes redundancy
#      masks = label_types.reshape(-1,1) == labels
#      clus = [np.nonzero(mask)[0] for mask in masks]
#      return clus
