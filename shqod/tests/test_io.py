"""Test IO functions."""

import unittest
import os

from shqod.io import (
    read_trajec_csv,
    trajecs_from_df,
    load_trajecs,
    load_trajecs_lex
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestIO(unittest.TestCase):
    """Test the function in the module io."""

    def setUp(self):
        """To be executed before running the other tests."""
        # the original data comes from Level 8, which has a grid of width 70
        self.grid_width = 70
        self.json_filenames = [
            os.path.join(THIS_DIR, 'testdata1.json'),
            os.path.join(THIS_DIR, 'testdata2.json')
        ]
        self.csv_filename = os.path.join(THIS_DIR, 'testdata.csv')
        df, n = read_trajec_csv(self.csv_filename, return_length=True)
        self.df = df
        self.n = n

    def test_load_trajecs(self):
        """Test reading JSON input (one file for each trajectory)."""
        trajec = load_trajecs(self.json_filenames)  # list of maps
        self.assertEqual(len(trajec), len(self.json_filenames))
        self.assertEqual(list(trajec[0]),
                         [(44, 10), (44, 10), (44, 11), (44, 11), (45, 11)])
        self.assertEqual(list(trajec[1]),
                         [(44, 10), (44, 11), (44, 11), (44, 11), (45, 11)])

    def test_load_trajecs_lex(self):
        """Test reading JSON input (one file for each trajectory)."""
        trajec = load_trajecs_lex(self.json_filenames, self.grid_width)
        self.assertEqual(len(trajec), len(self.json_filenames))
        self.assertEqual(list(trajec[0]), [744, 744, 814, 814, 815])
        self.assertEqual(list(trajec[1]), [744, 814, 814, 814, 815])

    def test_read_trajec_csv(self):
        """Test the reading function."""
        self.assertEqual(len(self.df), self.n)

    def test_trajecs_from_df(self):
        """Test loading trajectories from a csv file with JSON."""
        trajec = trajecs_from_df(self.df, lexico=False)
        self.assertEqual(list(next(trajec)),
                         [(44, 10), (44, 10), (44, 11), (44, 11), (45, 11)])
        self.assertEqual(list(next(trajec)),
                         [(44, 10), (44, 11), (44, 11), (44, 11), (45, 11)])

    def test_trajecs_from_df_lexico(self):
        """Test loading trajectories from a csv file with JSON (lexicographic)."""
        trajec = trajecs_from_df(self.df, lexico=True,
                                 grid_width=self.grid_width)
        self.assertEqual(list(next(trajec)), [744, 744, 814, 814, 815])
        self.assertEqual(list(next(trajec)), [744, 814, 814, 814, 815])


if __name__ == '__main__':
    unittest.main()
