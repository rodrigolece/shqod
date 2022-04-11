from typing import List, Dict, Union

import os
import re

import pandas as pd
import pyarrow.feather as feather

from shqod.io.read import read_path_csv, read_path_feather
from shqod.paths import vo_correctness


class LevelsLoader(object):
    """
    Loader for levels data stored together inside a single directory.

    The files are named following the convention:

        level_{level}_{country}_{gender}.{format}

    Currently we only match for `level` and `gender` (`country` is 'uk').
    Supported formats are csv and feather.

    Attributes
    ----------
    loaded : Dict[(level, gender), pd.DataFrame]
        The loaded DataFrames, indexed by level (int) and gender
        (one of 'f', 'm'). The DataFrames are loaded lazily by calls
        to the method `get`.

    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        fmt: str = "feather",
        path_col: str = "trajectory_data",
        raw: bool = False,
        suffix: str = None,
    ) -> Dict[str, str]:
        """
        Load the untidy CSV files contained within `directory`.

        Parameters
        ----------
        directory : str
        fmt : {'csv', 'feather'}
            Default is 'csv'.

        """
        if fmt not in ("csv", "feather"):
            raise ValueError(f"invalid format {fmt}")

        self.loaded = dict()
        self.norm_loaded = dict()
        self._files = dict()
        self._norm_files = dict()
        self._fmt = fmt
        self._path_col = path_col
        self._raw = raw

        for file in os.listdir(directory):
            pattern = rf"_(\d+)_\w*_(f|m){suffix if suffix else ''}.{fmt}"
            if match := re.search(pattern, file):
                lvl, gender = match.groups()
                key = (int(lvl), gender)
                self._files[key] = os.path.join(directory, file)

            norm_pattern = rf"^norm_uk_(f|m).{fmt}"
            if match := re.search(norm_pattern, file):
                gender = match.group(1)
                self._norm_files[gender] = os.path.join(directory, file)

        if self._norm_files:
            print("Normative correction found; call `get` with `norm=True`")

        return None

    def get(
        self,
        level: int,
        gender: str,
        age: Union[int, str] = None,
        filter_vo: bool = False,
        norm: bool = False,
        feat_types: List[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Get the DataFrame for a given level, gender and optionally age(s).

        The CSV files are loaded as needed for a given level and gender. The
        resulting DataFrames are stored in the attribute `loaded` and the
        selection by age is a simple filtering operation.

        Parameters
        ----------
        level : int
        gender : {'f', 'm'}
        age : int or str, optional
            If str should be provided as 'low:high' where low and high are
            inclusive.

        Returns
        -------
        pd.DataFrame

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

        key = (level, gender)
        file = self._files.get(key)

        if file:
            if key not in self.loaded:
                method = read_path_csv if self._fmt == "csv" else read_path_feather
                self.loaded[key] = method(file, self._path_col, raw=self._raw)
        else:
            raise FileNotFoundError

        df = self.loaded[key].copy()
        # Copying is handy because df can be modified inplace downstream

        # if age was passed as argument, filter
        if age is not None:
            idx = (df.age >= low) & (df.age <= high)
            df = df.loc[idx].reset_index(drop=True)

        if filter_vo:
            idx = vo_correctness(df.vo, level, verbose=verbose)
            df = df.loc[idx].reset_index(drop=True)

        if norm:
            if feat_types is None:
                raise ValueError("arg `feat_types` is needed when `norm` is `True`")
            df = self._norm_correction(level, gender, df, feat_types)

        return df

    def _norm_correction(self, level, gender, df, feat_types):
        if level in [1, 2]:
            raise ValueError("invalid level")

        file = self._norm_files.get(gender)

        if file:
            if gender not in self.norm_loaded:
                method = pd.read_csv if self._fmt == "csv" else feather.read_feather
                self.norm_loaded[gender] = method(file).set_index("id")
        else:
            raise FileNotFoundError

        out = df.copy().set_index("id")
        norm_df = self.norm_loaded[gender].reindex(out.index)
        out.loc[:, feat_types] = out[feat_types] / norm_df[feat_types]

        return out.reset_index()
