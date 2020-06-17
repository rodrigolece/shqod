"""Test IO functions."""

import unittest

from shqod.io import load_trajecs, load_trajecs_lex


class TestIO(unittest.TestCase):
    # def setUp(self):
    #     pass

    def test_load_trajecs(self):
        """Test reading JSON input (one file for each trajectory)."""
        self.assertEqual(load_trajecs(), [1, 0])  # TODO: Fill in real test

    def test_load_trajecs_lex(self):
        """Test reading JSON input (one file for each trajectory)."""
        self.assertEqual(load_trajecs_lex(), [1, 0])  # TODO: Fill in real test
