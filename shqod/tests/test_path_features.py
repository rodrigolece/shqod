"""Test path features."""

import unittest
import numpy as np
import scipy.sparse as sp

#  from shqod.io import (
#      read_path_csv,
#  )

from shqod.utils import sigmoid_ftn

from shqod.paths import (
    path_length,
    avg_curvature,
    bdy_affinity,
    #  fractal_dim,
    frobenius_deviation,
    supremum_deviation,
    sum_match,
    mobility_functional,
)


class TestPathFeatures(unittest.TestCase):
    """Test the functions in the paths.features module."""

    def setUp(self):
        self.path = np.array(
            [
                [1, 0],
                [1, 0],  # stationary point
                [1, 1],
                [0, 1],
                [-1, 1],
                [-1, 0],
            ]
        )
        self.bdy = np.array([[0, 0], [0, 1]])  # touches the boundary at (0, 1)

        self.small_path = np.array([[0, 0], [0, 1], [1, 1]])
        self.grid_size = (2, 2)
        self.od_mat = 0.1 * sp.csr_matrix(
            [
                [1, 8, 1, 1],
                [0, 0, 0, 8],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )

    def test_path_lenght(self):
        length = path_length(self.path)
        self.assertTrue(np.isclose(length, 4))

    def test_avg_curvature(self):
        curv = avg_curvature(self.path)
        self.assertTrue(np.isclose(curv, 2 * np.sqrt(2) / 6))

    def test_bdy_affinity(self):
        bdy = bdy_affinity(self.path, self.bdy, scale=0.5, rin=0, rout=1)
        expected_ds_rescaled = 0.5 * np.array([1, 1, 1, -1, 1, 1])
        vals = sigmoid_ftn(-expected_ds_rescaled)
        self.assertTrue(np.isclose(bdy, vals.sum() / 6))

    def test_froenius_deviation(self):
        pass

    def test_supremum_deviation(self):
        pass

    def test_sum_match(self):
        match = sum_match(self.small_path, self.grid_size, self.od_mat)
        self.assertTrue(np.isclose(match, -0.2 / 2))  # two matching entries

    def test_mobility_functional(self):
        diag = np.ones(2)
        diag *= 0.1 / np.linalg.norm(diag)
        right_up = np.array([0.8, 0.1])
        field = {
            (0, 0): right_up + diag,
            (0, 1): np.array([0.1, 0.0]),
            (1, 0): np.array([0.0, 0.8]),
        }
        mob = mobility_functional(self.small_path, field)
        dots = np.dot(field[(0, 0)], [0, 1]) + np.dot(field[(0, 1)], [1, 0])
        self.assertTrue(np.isclose(mob, -dots / 3))  # length 3


if __name__ == "__main__":
    unittest.main()
