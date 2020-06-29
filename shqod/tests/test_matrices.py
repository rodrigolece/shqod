"""Test matrix functions."""

import unittest
import numpy as np
import scipy.sparse as sp

from shqod.matrices import (
    od_matrix,
    reduce_matrix,
    calculate_field,
    field_to_dict
)


class TestIO(unittest.TestCase):
    """Test the functions in the module matrices."""

    def setUp(self):
        """To be executed before running the other tests."""
        self.grid_width = 3
        self.grid_length = 4  # assume last row is land
        self.lex_trajecs = [[0, 0, 1, 4, 5, 8], [0, 3, 4, 7, 8]]
        # note 0-> 0 (diagonal entry)

        # 2x2 data for field
        small_lex = [[0, 2, 3], [0, 1, 3], [0, 2, 3], [0, 1]]
        self.small_mat = od_matrix(small_lex, 4)

    def test_od_matrix(self):
        """Calculate OD matrix from trajectories."""
        n = self.grid_width * self.grid_length
        mat = od_matrix(self.lex_trajecs, n)

        self.assertTrue(sp.isspmatrix_csr(mat))
        self.assertEqual(mat.shape, (n, n))
        self.assertTrue(np.all(mat[np.diag_indices_from(mat)] == 0))

        i, j = mat.nonzero()
        self.assertEqual(i.tolist(), [0, 0, 1, 3, 4, 4, 5, 7])  # Os
        self.assertEqual(j.tolist(), [1, 3, 4, 4, 5, 7, 8, 8])  # Ds

    def test_reduce_matrix(self):
        """Eliminate zero rows and zero columns."""
        input = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        desired_out = np.array([[1, 0], [0, 1]])
        self.assertTrue(np.all(reduce_matrix(input) == desired_out))

    def test_field_to_dict(self):
        """Convert arrays that define field to dict."""
        Xs = np.array([[0, 0], [0, 1], [1, 0]])
        Fs = np.array([[1, 1], [1, 0], [0, 1]])
        d = field_to_dict(Xs, Fs)
        self.assertTrue(np.all(d[(0, 0)] == np.array([1, 1])))
        self.assertTrue(np.all(d[(0, 1)] == np.array([1, 0])))
        self.assertTrue(np.all(d[(1, 0)] == np.array([0, 1])))

    # def test_calculate_field(self):
    #     """Calculate the mobility field."""
    #     # [[0, 2, 3], [0, 1, 3], [0, 2, 3], [0, 1]]
    #     Xs, Fs = calculate_field(self.small_mat)
    #
    #     desired_out = ([(0, 0), (1, 0), (0, 1)], [(1, 1), (1, 0)])
    #
    #     self.assertEqual(calculate_field(), desired_out)

    def test_mobility_funcitonal(self):
        """Calculate a the mobility functional."""
        # TODO: Fill test
        pass


if __name__ == '__main__':
    unittest.main()
