"""Test IO functions."""

import unittest
import os

from shqod.io import (
    read_trajec_csv,
    trajecs_from_df,
    load_trajecs,
    load_trajecs_lex,
    load_level_grid
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestIO(unittest.TestCase):
    """Test the functions in the module io."""

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
        self.level_filename = os.path.join(THIS_DIR, 'testleveldata.json')

    def test_load_trajecs(self):
        """Load trajectories from JSON (one per file)."""
        trajec = load_trajecs(self.json_filenames)  # list of maps
        self.assertEqual(len(trajec), len(self.json_filenames))
        self.assertEqual(list(trajec[0]),
                         [(44, 10), (44, 10), (44, 11), (44, 11), (45, 11)])
        self.assertEqual(list(trajec[1]),
                         [(44, 10), (44, 11), (44, 11), (44, 11), (45, 11)])

    def test_load_trajecs_lex(self):
        """Load lexicographic trajectories from JSON (one per file)."""
        trajec = load_trajecs_lex(self.json_filenames, self.grid_width)
        self.assertEqual(len(trajec), len(self.json_filenames))
        self.assertEqual(list(trajec[0]), [744, 744, 814, 814, 815])
        self.assertEqual(list(trajec[1]), [744, 814, 814, 814, 815])

    def test_read_trajec_csv(self):
        """Read trajectories csv."""
        self.assertEqual(len(self.df), self.n)

    def test_trajecs_from_df(self):
        """Load trajectories from DataFrame."""
        trajec = trajecs_from_df(self.df, lexico=False)
        self.assertEqual(list(next(trajec)),
                         [(44, 10), (44, 10), (44, 11), (44, 11), (45, 11)])
        self.assertEqual(list(next(trajec)),
                         [(44, 10), (44, 11), (44, 11), (44, 11), (45, 11)])

    def test_trajecs_from_df_lexico(self):
        """Load lexicographic trajectories from DataFrame."""
        trajec = trajecs_from_df(self.df, lexico=True,
                                 grid_width=self.grid_width)
        self.assertEqual(list(next(trajec)), [744, 744, 814, 814, 815])
        self.assertEqual(list(next(trajec)), [744, 814, 814, 814, 815])

    def test_load_level_grid(self):
        """Load grid data."""
        x, y, width, length = load_level_grid(self.level_filename)
        self.assertEqual(width, 5)
        self.assertEqual(length, 4)
        self.assertEqual(len(x), 15)


if __name__ == '__main__':
    unittest.main()
