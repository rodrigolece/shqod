"""Main functions to build an OD matrix and calculate the field."""

from typing import Iterable, Tuple, Dict
from .dtypes import Trajec, LexTrajec

import numpy as np
import scipy.sparse as sp


def od_matrix(lex_trajecs: Iterable[LexTrajec],
              grid_size: int) -> sp.csr.csr_matrix:
    """Calculate the OD matrix from a set lexicographic trajectories.

    Parameters
    ----------
    lex_trajecs : Iterable[LexTrajec]
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
        lst = list(t)

        for i, j in zip(lst[:-1], lst[1:]):
            out[i, j] += 1

    # Remove diagonal entries
    out[np.diag_indices_from(out)] = 0
    csr = out.tocsr()
    csr.eliminate_zeros()

    return csr


def reduce_matrix(square_mat: sp.csr.csr_matrix,
                  return_index: bool = False) -> sp.csr.csr_matrix:
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


def calculate_field(od_mat: sp.csr.csr_matrix,
                    grid_width: int,
                    nb_trajecs: int = None) -> Tuple[np.array, np.array]:
    """Calculate the field at each location (origin) where it is non-zero.

    Parameters
    ----------
    od_mat : csr_matrix
        The orgin-destination (OD) matrix to use as input.
    grid_width : int
        The width of the grid in the level.
    nb_trajecs : int or None, optional
        If provided, normalise the field by dividing by this number.

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

    if nb_trajecs:
        Fs /= nb_trajecs

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


def mobility_functional(trajec: Trajec,
                        field: Dict[Tuple[int, int], np.array]) -> float:
    """Short summary.

    Parameters
    ----------
    trajec : Trajec
        The (x, y) trajectory for which the functional is computed.
    field : Dict[Tuple[int, int], np.array]
        The mobility field used in the comparison.

    Returns
    -------
    float
        The mobility functional.

    """
    # TODO: define field dtype and put in .dtypes
    out = 0.0
    N = 0

    for el in trajec:
        Fi = field.get(tuple(el), np.zeros(2))
        out += np.dot(Fi, el)
        N += 1

    return out / N
