"""Test IO functions."""

import unittest
import os
from pathlib import Path

from shqod.io import (
    read_path_csv,
    duplicated_attempts,
    paths_from_df,
    paths_from_files,
    read_level_grid,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dir = Path(THIS_DIR)


class TestIO(unittest.TestCase):
    """Test the functions in the module io."""

    def setUp(self):
        """To be executed before running the other tests."""
        # the original data comes from Level 8, which has a grid of width 70
        self.grid_width = 70
        self.json_filenames = [load_dir / f for f in ("data1.json", "data2.json")]
        self.csv_filename = load_dir / "data.csv"
        #  df = read_path_csv(self.csv_filename, path_col="trajectory_data", raw=False)
        #  self.df = df
        self.level_filename = load_dir / "data_level.json"

    def test_paths_from_files(self):
        """Load paths from JSON (one per file)."""
        path = paths_from_files(self.json_filenames)
        self.assertEqual(
            next(path).tolist(), [[44, 10], [44, 10], [44, 11], [44, 11], [45, 11]]
        )
        self.assertEqual(
            next(path).tolist(), [[44, 10], [44, 11], [44, 11], [44, 11], [45, 11]]
        )

    def test_paths_from_files_lexico(self):
        """Load lexicographic paths from JSON (one per file)."""
        path = paths_from_files(
            self.json_filenames, lexico=True, grid_width=self.grid_width
        )
        self.assertEqual(next(path).tolist(), [744, 744, 814, 814, 815])
        self.assertEqual(next(path).tolist(), [744, 814, 814, 814, 815])

    def test_read_path_csv(self):
        """Read paths csv."""
        df = read_path_csv(self.csv_filename, path_col="trajectory_data", raw=False)
        paths = df["trajectory_data"]

        self.assertEqual(
            paths.iloc[0].tolist(), [[44, 10], [44, 10], [44, 11], [44, 11], [45, 11]]
        )
        self.assertEqual(
            paths.iloc[1].tolist(), [[44, 10], [44, 11], [44, 11], [44, 11], [45, 11]]
        )
        # I've added a duplicated line (2nd attempt)
        self.assertIsNone(paths.iloc[2])
        self.assertEqual(
            paths.iloc[3].tolist(), [[44, 10], [44, 11], [44, 11], [44, 11], [45, 11]]
        )

    def test_paths_from_df_lexico(self):
        """Load lexicographic paths from DataFrame."""
        df = read_path_csv(self.csv_filename, path_col="trajectory_data", raw=True)
        paths = paths_from_df(df, lexico=True, grid_width=self.grid_width)
        self.assertEqual(next(paths).tolist(), [744, 744, 814, 814, 815])
        self.assertEqual(next(paths).tolist(), [744, 814, 814, 814, 815])
        # I've added a duplicated line (2nd attempt)
        self.assertEqual(next(paths).tolist(), [744, 814, 814, 814, 815])

    def test_duplicated_attempts_keep_first(self):
        """Test dropping all but first attempt for each player."""
        df = read_path_csv(self.csv_filename, path_col="trajectory_data", raw=True)
        idx = duplicated_attempts(df, keep="first")
        filtered_df = df.loc[idx]
        self.assertEqual(len(filtered_df), len(df) - 1)

    def test_duplicated_attempts_keep_last(self):
        """Test dropping all but last attempt for each player."""
        df = read_path_csv(self.csv_filename, path_col="trajectory_data", raw=True)
        idx = duplicated_attempts(df, keep="last")
        filtered_df = df.loc[idx]
        self.assertEqual(len(filtered_df), len(df) - 1)

    def test_read_level_grid(self):
        """Read grid data."""
        coords, width, length = read_level_grid(self.level_filename)
        self.assertEqual(width, 5)
        self.assertEqual(length, 4)
        self.assertEqual(coords.shape, (15, 2))


if __name__ == "__main__":
    unittest.main()
