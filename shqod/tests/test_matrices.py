"""Test matrix functions."""

import unittest
import numpy as np
import scipy.sparse as sp

from shqod.matrices import (
    od_matrix,
    mobility_functional,
    calculate_field,
    field_to_dict,
)


class TestMatrices(unittest.TestCase):
    """Test the functions in the module matrices."""

    def setUp(self):
        """To be executed before running the other tests."""
        self.grid_size = 2, 2  # width, length

        one = np.array([[0, 0], [0, 0], [0, 1], [1, 1]])  # note 0-> 0 (diagonal entry)
        two = np.array([[0, 0], [1, 0], [1, 1]])
        three = np.array([[0, 0], [0, 1], [1, 1]])
        four = np.array([[0, 0], [1, 0]])
        self.paths = [one, two, three, four]
        #  self.paths = [[0, 0, 2, 3], [0, 1, 3], [0, 2, 3], [0, 1]]

        self.desired_od = np.array(
            [
                [0, 2, 2, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 0, 0],
            ]
        )

    def test_od_matrix(self):
        """Calculate OD matrix from paths."""
        grid_size = self.grid_size
        mat = od_matrix(self.paths, grid_size, remove_diag=False)
        n = np.multiply(*grid_size)

        self.assertTrue(sp.isspmatrix_csr(mat))
        self.assertEqual(mat.shape, (n, n))

        mat = od_matrix(self.paths, grid_size, remove_diag=True)
        self.assertTrue(np.all(mat[np.diag_indices_from(mat)] == 0))
        self.assertTrue(np.allclose(mat.toarray(), self.desired_od))

    def test_calculate_field(self):
        """Calculate the mobility field."""
        width = self.grid_size[0]
        mat = sp.csr_matrix(self.desired_od)
        Xs, Fs = calculate_field(mat, width)

        # Note that test is susceptible to order of arrays
        desired_Xs = [[0, 0], [1, 0], [0, 1]]
        desired_Fs = [[2, 2], [0, 1], [2, 0]]

        self.assertTrue(np.allclose(Xs, desired_Xs))
        self.assertTrue(np.allclose(Fs, desired_Fs))

    def test_field_to_dict(self):
        """Convert arrays that define field to dict."""
        Xs = np.array([[0, 0], [1, 0], [0, 1]])
        Fs = np.array([[2, 2], [0, 1], [2, 0]])

        d = field_to_dict(Xs, Fs)
        self.assertTrue(np.allclose(d[(0, 0)], [2, 2]))
        self.assertTrue(np.allclose(d[(1, 0)], [0, 1]))
        self.assertTrue(np.allclose(d[(0, 1)], [2, 0]))

    def test_mobility_functional(self):
        """Calculate the mobility functional."""
        # TODO: Fill test
        # TODO: move to paths.features
        pass


if __name__ == "__main__":
    unittest.main()
