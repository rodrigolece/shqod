"""Test IO functions."""

import unittest
import os

from shqod.io import (
    read_trajec_csv,
    duplicated_attempts,
    trajecs_from_df,
    trajecs_from_files,
    read_level_grid,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestIO(unittest.TestCase):
    """Test the functions in the module io."""

    def setUp(self):
        """To be executed before running the other tests."""
        # the original data comes from Level 8, which has a grid of width 70
        self.grid_width = 70
        self.json_filenames = [
            os.path.join(THIS_DIR, "testdata1.json"),
            os.path.join(THIS_DIR, "testdata2.json"),
        ]
        self.csv_filename = os.path.join(THIS_DIR, "testdata.csv")
        df = read_trajec_csv(self.csv_filename)
        self.df = df
        self.n = len(df)
        self.level_filename = os.path.join(THIS_DIR, "testleveldata.json")

    def test_trajecs_from_files(self):
        """Load trajectories from JSON (one per file)."""
        trajec = trajecs_from_files(self.json_filenames)
        self.assertEqual(
            next(trajec).tolist(), [[44, 10], [44, 10], [44, 11], [44, 11], [45, 11]]
        )
        self.assertEqual(
            next(trajec).tolist(), [[44, 10], [44, 11], [44, 11], [44, 11], [45, 11]]
        )

    def test_trajecs_from_files_lexico(self):
        """Load lexicographic trajectories from JSON (one per file)."""
        trajec = trajecs_from_files(
            self.json_filenames, lexico=True, grid_width=self.grid_width
        )
        self.assertEqual(next(trajec).tolist(), [744, 744, 814, 814, 815])
        self.assertEqual(next(trajec).tolist(), [744, 814, 814, 814, 815])

    def test_read_trajec_csv(self):
        """Read trajectories csv."""
        self.assertEqual(len(self.df), self.n)

    def test_trajecs_from_df(self):
        """Load trajectories from DataFrame."""
        trajec = trajecs_from_df(self.df, lexico=False)
        self.assertEqual(
            next(trajec).tolist(), [[44, 10], [44, 10], [44, 11], [44, 11], [45, 11]]
        )
        self.assertEqual(
            next(trajec).tolist(), [[44, 10], [44, 11], [44, 11], [44, 11], [45, 11]]
        )
        # I've added a duplicated line (2nd attempt)
        self.assertEqual(
            next(trajec).tolist(), [[44, 10], [44, 11], [44, 11], [44, 11], [45, 11]]
        )

    def test_trajecs_from_df_lexico(self):
        """Load lexicographic trajectories from DataFrame."""
        trajec = trajecs_from_df(self.df, lexico=True, grid_width=self.grid_width)
        self.assertEqual(next(trajec).tolist(), [744, 744, 814, 814, 815])
        self.assertEqual(next(trajec).tolist(), [744, 814, 814, 814, 815])
        # I've added a duplicated line (2nd attempt)
        self.assertEqual(next(trajec).tolist(), [744, 814, 814, 814, 815])

    def test_duplicated_attempts_keep_first(self):
        """Test dropping all but first attempt for each player."""
        idx = duplicated_attempts(self.df, keep="first")
        filtered_df = self.df.loc[idx]
        self.assertEqual(len(filtered_df), len(self.df) - 1)

    def test_duplicated_attempts_keep_last(self):
        """Test dropping all but last attempt for each player."""
        idx = duplicated_attempts(self.df, keep="last")
        filtered_df = self.df.loc[idx]
        self.assertEqual(len(filtered_df), len(self.df) - 1)

    def test_read_level_grid(self):
        """Read grid data."""
        coords, width, length = read_level_grid(self.level_filename)
        self.assertEqual(width, 5)
        self.assertEqual(length, 4)
        self.assertEqual(coords.shape, (15, 2))


if __name__ == "__main__":
    unittest.main()
