"""Main functions to build an OD matrix and calculate the field."""

from typing import Iterable, Tuple
from .dtypes import LexTrajec

import numpy as np
import scipy
import scipy.sparse as sp


def od_matrix(lex_trajecs: Iterable[LexTrajec],
              grid_size: int) -> scipy.sparse.csr.csr_matrix:
    """Calculate the OD matrix from a set lexicographic trajectories.

    Parameters
    ----------
    lex_trajecs : Iterable[LexTrajec]
        The lexicographic trajectories to be used in counting the number of
        trips between locations.
    grid_size : int
        The size of the grid in the level calculated as `width * height`.

    Returns
    -------
    scipy.sparse.csr.csr_matrix
        Origin-destination (OD) matrix.

    """
    out = sp.lil_matrix((grid_size, grid_size), dtype=int)

    for t in lex_trajecs:
        lst = list(t)

        for i, j in zip(lst[:-1], lst[1:]):
            out[i, j] += 1

    # Remove diagonal entries
    out[np.diag_indices_from(out)] = 0

    return out.tocsr()


def reduce_matrix(square_mat: scipy.sparse.csr.csr_matrix,
                  return_zeros: bool = False) -> scipy.sparse.csr.csr_matrix:
    """Remove all rows and columns that are simultaneously empty.

    Parameters
    ----------
    square_mat : csr_matrix
        Input matrix.
    return_zeros : bool
        If True, also return the indices of `square_mat` that were removed.

    Returns
    -------
    reduced_mat : csr_matrix
        Description of returned object.
    idx : ndarray, optional

    """
    i, j = square_mat.nonzero()
    idx = sorted(set(i).union(set(j)))  # sorted converts to list
    reduced_mat = square_mat[idx, :][:, idx]

    return (reduced_mat, idx) if return_zeros else reduced_mat


def calculate_field(od_mat: scipy.sparse.csr.csr_matrix,
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
