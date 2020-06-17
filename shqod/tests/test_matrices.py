"""Test matrix functions."""

import unittest
import numpy as np
import scipy.sparse as sp

from shqod.matrices import (
    od_matrix,
    reduced_mat,
    calculate_field
)


class TestIO(unittest.TestCase):
    # def setUp(self):
    #     pass

    def test_od_matrix(self):
        """Test calculating an OD matrix from trajectories."""
        mat = od_matrix()
        self.assertIsInstance(mat,  sp.csr.csr_matrix)  # TODO: Fill in real test

        desired_out = np.array([[1, 0], [0, 1]])
        self.assertEqual(mat, desired_out)  # TODO: Fill in real test

    def test_reduce_matrix(self):
        """Test elimanating zero rows and zero columns."""
        input = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        desired_out = np.array([[1, 0], [0, 1]])
        # TODO: Fill in real test
        self.assertEqual(reduced_mat(input), desired_out)

    def test_calculate_field(self):
        """Test the calculation of the mobility field."""
        desired_out = ([(0, 0), (1, 0)], [(1, 0), (1, 0)])
        # TODO: Fill in real test
        self.assertEqual(calculate_field(), desired_out)
