"""Main functions to build an OD matrix and calculate the field."""

import warnings
from typing import Iterable, List, Tuple, Dict

# from .dtypes import Trajec, LexTrajec

import numpy as np
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from itertools import groupby
from operator import itemgetter


def od_matrix(lex_trajecs: Iterable[np.array], grid_size: int) -> sp.csr.csr_matrix:
    """Calculate the OD matrix from a set lexicographic trajectories.

    Parameters
    ----------
    lex_trajecs : Iterable[np.array]
        The lexicographic trajectories to be used in counting the number of
        trips between locations.
    grid_size : int
        The size of the grid in the level calculated as `width * length`.

    Returns
    -------
    sp.csr.csr_matrix
        Origin-destination (OD) matrix.

    """
    out = sp.lil_matrix((grid_size, grid_size), dtype=int)

    for t in lex_trajecs:
        for i, j in zip(t[:-1], t[1:]):
            out[i, j] += 1

    # Remove diagonal entries
    out[np.diag_indices_from(out)] = 0
    csr = out.tocsr()
    csr.eliminate_zeros()

    return csr


def od_matrix_brokenup(
    trajecs: Iterable[np.array],
    grid_size: Tuple[int, int],
    flags: np.array,
    R: float = 3,
) -> List[sp.csr.csr_matrix]:
    """Calculate the OD matrices broken up by the visits to the checkpoints.

    Parameters
    ----------
    trajecs : Iterable[np.array]
        The trajectories to be used in counting the number of
        trips between locations.
    grid_size : int
        The size of the grid in the level calculated as `width * length`.
    flags : np.array
        The coordinates of the flags (ordered).
    R : float
        The radius to consider a checkpoint visited.

    Returns
    -------
    List[sp.csr.csr_matrix]
        List of origin-destination (OD) matrices.

    """
    N = len(flags)
    width, length = grid_size
    grid_size = width * length
    out = [sp.lil_matrix((grid_size, grid_size), dtype=int) for _ in range(N)]

    for arr in trajecs:
        idx = breakup_array_by_flags(arr, flags, R=3)

        for k, sub_arr in enumerate(np.split(arr, idx[:-1])):
            lex = sub_arr[:, 1] * width + sub_arr[:, 0]

            for i, j in zip(lex[:-1], lex[1:]):
                out[k][i, j] += 1

    # Remove diagonal entries
    for k in range(N):
        out[k][np.diag_indices_from(out[k])] = 0
        out[k] = out[k].tocsr()
        out[k].eliminate_zeros()

    return out


def breakup_array_by_flags(
    trajec: np.array, flags: np.array, R: float = 3
) -> List[int]:
    """Find the last index of the first passage by each flag.

    Parameters
    ----------
    trajec : np.array
        An array storing the trajectory.
    flags : np.array
        The coordinates of the flags (ordered).
    R : float
        The radius to consider a checkpoint visited.

    Returns
    -------
    List[int]
        The list with the indices that can be used to split `trajec`.

    """
    out = []
    offset = 0

    for f in flags:
        idx = np.where(np.linalg.norm(trajec[offset:] - f, 2, axis=1) <= R)[0] + offset
        max_sets = [
            max(map(itemgetter(1), g))
            for _, g in groupby(enumerate(idx), lambda ix: ix[0] - ix[1])
        ]
        offset = min(max_sets)
        out.append(offset)

    return out


def reduce_matrix(
    square_mat: sp.csr.csr_matrix, return_index: bool = False
) -> sp.csr.csr_matrix:
    """Remove all rows and columns that are simultaneously empty.

    Parameters
    ----------
    square_mat : csr_matrix
        Input matrix.
    return_index : bool
        If True, also return the indices of `square_mat` that were kept.

    Returns
    -------
    reduced_mat : sp.csr_matrix
        A square matrix that has the zero rows and columns removed.
    idx : np.array, optional
        The index of the rows and columns that were kept

    """
    i, j = square_mat.nonzero()
    idx = sorted(set(i).union(set(j)))  # sorted converts to list
    reduced_mat = square_mat[idx, :][:, idx]

    return (reduced_mat, idx) if return_index else reduced_mat


def calculate_field(
    od_mat: sp.csr.csr_matrix, grid_width: int
) -> Tuple[np.array, np.array]:
    """Calculate the field at each location (origin) where it is non-zero.

    Parameters
    ----------
    od_mat : csr_matrix
        The orgin-destination (OD) matrix to use as input.
    grid_width : int
        The width of the grid in the level.

    Returns
    -------
    Xs : np.array
        The coordinates of the non-zero entries of the field.
    Fs : np.array
        The entries of the field.

    """
    i, j = od_mat.nonzero()
    r_origin = np.vstack((i % grid_width, i // grid_width)).T
    r_destination = np.vstack((j % grid_width, j // grid_width)).T

    u_vec = r_destination - r_origin
    u_vec = u_vec / np.linalg.norm(u_vec, axis=1)[:, np.newaxis]

    weighted_vec = u_vec * np.asarray(od_mat[i, j]).T

    # We sum the vectors grouped by origin using counts of unique elements
    _, idx_uniq, counts = np.unique(i, return_index=True, return_counts=True)
    c = np.cumsum(counts)  # the end index of each group

    nb_locations = len(idx_uniq)
    Xs = r_origin[idx_uniq]  # The position at which the field is non-zero

    Fs = np.zeros((nb_locations, 2))  # Pre-allocate memory for the field

    start = 0
    for k, end in enumerate(c):
        Fs[k] += weighted_vec[start:end].sum(axis=0)
        start = end

    return Xs, Fs


def field_to_dict(Xs: np.array, Fs: np.array) -> Dict[Tuple, np.array]:
    """Convert array summarising Field into dict.

    Parameters
    ----------
    Xs : np.array
        The array to use as keys (stores as rows).
    Fs : np.array
        The values of the field that will be dict values.

    Returns
    -------
    Dict[Tuple, np.array]
        A dictionary that is indixed by the values in Xs, and whose values are
        the Fs.

    """
    out = dict()
    for k in range(len(Xs)):
        out[tuple(Xs[k])] = Fs[k]

    return out


def visiting_order(
    trajec: np.array, flags: np.array, R: int = 3, safe_mode: bool = False
) -> np.array:

    """Calculate the order in which the flags are visited.

    Calling this function with the smoothened trajectories will be inefficient.

    Parameters
    ----------
    trajec : np.array
        The (x, y) trajectory for which the visiting order is calculated.
    flags : np.array
        The (x, y) coordinates of the flags (with the right order).
    R : float, optional
        The radius around which the trajectory is considered to have circled
        the flags.
    safe_mode : bool, optional
        Test the fist and the last flag (default is False).

    Returns
    -------
    Iterable
        The order in which the flags are actually visited.

    """
    dmat = distance_matrix(trajec, flags)
    _, j = (dmat < R).nonzero()
    out = [x[0] for x in groupby(j)]

    if safe_mode and out[0] != 0:
        warnings.warn("unexpeted first flag")
    if safe_mode and out[-1] != len(flags) - 1:
        warnings.warn("unexpected last flag")

    return out


def mobility_functional(
    trajec: np.array, od_mat: sp.csr.csr_matrix, grid_width: int, flags: np.array = None
) -> float:
    """Short summary.

    Parameters
    ----------
    trajec : np.array
        The (x, y) trajectory for which the functional is computed.
    od_mat : csr_matrix
        The orgin-destination (OD) matrix to use as input.
    grid_width : int
        The width of the grid in the level.
    flags : np.array, optional
        The (x, y) coordinates of the flags (with the right order).

    Returns
    -------
    float
        The mobility functional.

    """
    out = 0.0
    N = 0

    if isinstance(od_mat, list):
        assert flags is not None, "error: provide the coodinates of the flags"
        field = [field_to_dict(*calculate_field(mat, grid_width)) for mat in od_mat]

        idx = breakup_array_by_flags(trajec, flags, R=3)

        for k, sub_arr in enumerate(np.split(trajec, idx[:-1])):
            for el in sub_arr:
                Fi = field[k].get(tuple(el), np.zeros(2))
                out += np.dot(Fi, el)
                N += 1

    else:
        field = field_to_dict(*calculate_field(od_mat, grid_width))

        for el in trajec:
            Fi = field.get(tuple(el), np.zeros(2))
            out += np.dot(Fi, el)
            N += 1

    return out / N
