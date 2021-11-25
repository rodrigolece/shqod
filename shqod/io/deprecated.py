"""Read and write data."""

from typing import Tuple, Dict, Iterable, Union

import os
import re

import pandas as pd


class TidyLoader(object):
    """
    Loader for tidy data.

    Tidy data has the different levels for a given participant stored as
    different rows within a CSV file. We have one file for each group of
    participants (e.g. genetic group, diagnosed with AD, etc). Tidy data is
    useful for a small group (<100) of participants.

    Attributes
    ----------
    loaded : Dict[str, pd.DataFrame]
        The loaded path DataFrames, indexed by group keys (built from
        the names of the files by removing their extension).

    """

    def __init__(self, *dirs, path_col: str = "trajectory_data", raw: bool = False):
        """
        Load (a) directories(y) containing tidy CSV files.

        Parameters
        ----------
        *dirs : Variable length argument list
            The list of directories to load

        """
        self.loaded = dict()
        self._path_col = path_col
        self._raw = raw

        for d in dirs:
            self.loaded.update(self._load_dir(d))

        return None

    def _load_dir(self, directory: str) -> Dict[str, pd.DataFrame]:
        """Load csv files contained inside `directory`."""
        files = [os.path.join(directory, x) for x in os.listdir(directory)]

        out = dict()

        for f in files:
            df = read_path_csv(f, self._path_col, self._raw)
            key = os.path.basename(f).replace(".csv", "")
            out[key] = df

        return out

    def get(
        self,
        level: int,
        gender: str = None,
        age: Union[int, str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get the portion of the DataFrames for a level and optionally gender.

        Parameters
        ----------
        level : int
        gender : {'f', 'm'}, optional

        Returns
        -------
        Dict[str, pd.DataFrame]

        """
        assert type(age) in (int, str) or age is None, "invalid type for age"

        if isinstance(age, str):
            pattern = r"^(\d*):(\d*)$"
            match = re.search(pattern, age)
            if match:
                low, high = match.groups()
                low = 0 if low == "" else int(low)
                high = 99 if high == "" else int(high)
            else:
                raise ValueError("invalid format for age")

        elif isinstance(age, int):
            low, high = age, age

        out = dict()

        for key, df in self.loaded.items():
            idx = df.level == level
            if gender:
                idx = idx & (df.gender == gender)

            if age is not None:  # if age was passed as an argument
                idx = idx & (df.age >= low) & (df.age <= high)

            if idx.any():
                out[key] = df.loc[idx]

        return out
