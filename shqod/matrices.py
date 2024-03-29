"""Main functions to build an OD matrix and calculate the field."""

from typing import Iterable, List, Tuple, Dict
from itertools import groupby
from operator import itemgetter

import numpy as np
import scipy.sparse as sp


def od_matrix(
    paths: Iterable[np.ndarray],
    grid_size: Tuple[int, int],
    remove_diag: bool = False,
) -> sp.csr_matrix:
    """
    Calculate the OD matrix from a set lexicographic paths.

    Parameters
    ----------
    paths : Iterable[np.ndarray]
        The paths to be used in counting the number of trips between locations.
    grid_size : Tuple[int, int]
        The size of the grid (width, length).
    remove_diag : bool, optional
        Delete the entries in the diagonal (default is False).

    Returns
    -------
    sp.csr_matrix
        Origin-destination (OD) matrix.

    """
    width, _ = grid_size
    size = np.multiply(*grid_size)
    out = sp.lil_matrix((size, size), dtype=int)

    for t in paths:
        lex = width * t[:, 1] + t[:, 0]

        for i, j in zip(lex[:-1], lex[1:]):
            out[i, j] += 1

    if remove_diag:
        out[np.diag_indices_from(out)] = 0

    return out.tocsr()


def od_matrix_brokenup(
    paths: Iterable[np.ndarray],
    grid_size: Tuple[int, int],
    flags: np.ndarray,
    R: float = 3,
    remove_diag: bool = False,
) -> List[sp.csr_matrix]:
    """
    Calculate the OD matrices broken up by the visits to the checkpoints.

    Parameters
    ----------
    paths : Iterable[np.ndarray]
        The paths to be used in counting the number of trips between locations.
    grid_size : Tuple[int, int]
        The size of the grid (width, length).
    flags : np.ndarray
        The coordinates of the flags (ordered).
    R : float
        The radius to consider a checkpoint visited.
    remove_diag : bool, optional
        Delete the entries in the diagonal (default is False).

    Returns
    -------
    List[sp.csr_matrix]
        List of origin-destination (OD) matrices.

    """
    width, _ = grid_size
    size = np.multiply(*grid_size)

    nb_flags = len(flags)
    out = [sp.lil_matrix((size, size), dtype=int) for _ in range(nb_flags)]

    for arr in paths:
        idx = breakup_by_flags(arr, flags, R=R)

        for k, sub_arr in enumerate(np.split(arr, idx[:-1])):
            lex = sub_arr[:, 1] * width + sub_arr[:, 0]

            for i, j in zip(lex[:-1], lex[1:]):
                out[k][i, j] += 1

    for k in range(nb_flags):
        out[k] = out[k].tocsr()

        if remove_diag:
            out[k][np.diag_indices_from(out[k])] = 0

    return out


def breakup_by_flags(path: np.ndarray, flags: np.ndarray, R: float = 3) -> List[int]:
    """
    Find the last index of the first passage by each flag.

    Parameters
    ----------
    path : np.ndarray
        An array storing the path.
    flags : np.ndarray
        The coordinates of the flags (ordered).
    R : float
        The radius to consider a checkpoint visited.

    Returns
    -------
    List[int]
        The list with the indices that can be used to split `path`.

    """
    out = []
    offset = 0

    for f in flags:
        idx = np.where(np.linalg.norm(path[offset:] - f, 2, axis=1) <= R)[0] + offset
        max_sets = [
            max(map(itemgetter(1), g))
            for _, g in groupby(enumerate(idx), lambda ix: ix[0] - ix[1])
        ]
        offset = min(max_sets)
        out.append(offset)

    return out


def mobility_field(
    od_mat: sp.csr_matrix, grid_width: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Calculate the field at each location (origin) where it is non-zero.

    Parameters
    ----------
    od_mat : csr_matrix
        The orgin-destination (OD) matrix to use as input.
    grid_width : int
        The width of the grid in the level.

    Returns
    -------
    Dict[Tuple, np.ndarray]
        A dictionary representing the field indixed by the non-zero entries.

    """
    if not sp.issparse(od_mat):
        raise ValueError("invalid type")

    i, j = od_mat.nonzero()
    r_origin = np.vstack((i % grid_width, i // grid_width)).T
    r_destination = np.vstack((j % grid_width, j // grid_width)).T
    u_vec = (r_destination - r_origin).astype(float)

    # normalize non-zero entries
    norm = np.linalg.norm(u_vec, axis=1)
    idx = norm > 0
    u_vec[idx] = u_vec[idx] / norm[idx, np.newaxis]

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

    out = dict()
    for k in range(len(Xs)):
        out[tuple(Xs[k])] = Fs[k]

    return out
