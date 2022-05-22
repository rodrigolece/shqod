from typing import List, Dict, Union

import os
from pathlib import Path
import re

import pandas as pd
import pyarrow.feather as feather

from shqod.io.read import read_path_csv, read_path_feather
from shqod.paths import vo_correctness
from shqod.utils import parse_age


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
        country: str = "uk",
    ) -> Dict[str, str]:
        """
        Load the untidy CSV files contained within `directory`.

        Parameters
        ----------
        directory : str
        fmt : {'csv', 'feather'}
            Default is 'csv'.

        """
        directory = Path(directory)

        if not directory.is_dir():
            raise ValueError("not a directory")

        elif fmt not in ("csv", "feather"):
            raise ValueError(f"invalid format {fmt}")

        self._country = country
        self.loaded = dict()
        self._files = dict()
        self._norm_file = None
        self.norm_loaded = None
        self._fmt = fmt
        self._path_col = path_col
        self._raw = raw

        for file in os.listdir(directory):
            pattern = rf"^level_(\d+)_(\w*)_(f|m){suffix if suffix else ''}.{fmt}"
            if match := re.search(pattern, file):
                lvl, country, gender = match.groups()
                key = (int(lvl), gender)
                self._files[key] = os.path.join(directory, file)

            norm_pattern = rf"^norm_{self._country}.{fmt}"
            if match := re.search(norm_pattern, file):
                self._norm_file = os.path.join(directory, file)

        if self._norm_file:
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
            low, high = parse_age(age)
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
            df = self._norm_correction(level, df, feat_types)

        return df

    def _norm_correction(self, level, df, feat_types):
        if level in [1, 2]:
            raise ValueError("invalid level")

        file = self._norm_file

        if file:
            if self.norm_loaded is None:
                method = pd.read_csv if self._fmt == "csv" else feather.read_feather
                norm_df = method(file).set_index("id")
                assert norm_df.shape[1] == 2
                # The file should have only two columns: gender and the values
                col = list(set(norm_df.columns).difference(["gender"]))[0]
                self.norm_loaded = norm_df[col]
        else:
            raise FileNotFoundError

        out = df.copy().set_index("id")
        norm_factor = self.norm_loaded.reindex(out.index)
        out.loc[:, feat_types] = out[feat_types].divide(norm_factor, axis=0)

        return out.reset_index()
