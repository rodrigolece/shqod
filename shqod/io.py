"""Read and write data."""

from typing import Tuple, Dict, Iterable, Union

import os
import warnings
import re
import json
import collections
import pyarrow.feather as feather
import numpy as np
import pandas as pd

from .utils import vo_correctness


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


class UntidyLoader(object):
    """
    Loader for untidy data.

    Untidy data has separate CSV files for different levels and the data for
    a given participant is spread accross them. Untidy data is useful for a
    large group (>10,000) of participants, particularly when keeping track of
    the individuals is not the main priority.

    The files are named following the convention:

        level_{level}_{country}_{gender}.csv

    Currently we only match for `level` and `gender` (`country` is 'uk')

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
        fmt: str = "csv",
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
        self._files = dict()
        self._fmt = fmt
        self._path_col = path_col
        self._raw = raw

        for file in os.listdir(directory):
            pattern = rf"_(\d+)_\w*_(f|m){suffix if suffix is not None else ''}.{fmt}"
            match = re.search(pattern, file)
            if match:
                lvl, gender = match.groups()
                key = (int(lvl), gender)
                self._files[key] = os.path.join(directory, file)

        return None

    def get(
        self,
        level: int,
        gender: str,
        age: Union[int, str] = None,
        filter_vo: bool = False,
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

        return df


def convert_json_to_array(path_col: pd.Series) -> pd.Series:
    """Convert a column from json to array.

    Parameters
    ----------
    path_col : pd.Series

    """

    out = []

    for i, row in enumerate(path_col):
        t = path(row)

        if t is None:
            warnings.warn(f"corrupted data for entry: {i}")

        out.append(t)

    return pd.Series(out)


def read_path_csv(filename: str, path_col: str, raw: bool = False) -> pd.DataFrame:
    """
    Read a csv file containing path data into a DataFrame.

    Parameters
    ----------
    filename : str
    path_col : str
        Name of the column taht contains path data.
    raw : bool, optional
        Default is False

    Returns
    -------
    pd.DataFrame

    """
    df = pd.read_csv(filename)
    series = df.get(path_col)

    if series is not None and not raw:
        if isinstance(series.iloc[0], str):
            df[path_col] = convert_json_to_array(series)

    return df


def read_path_feather(filename: str, path_col: str, raw: bool = False) -> pd.DataFrame:
    """
    Read a feather file containing path data into a DataFrame.

    Parameters
    ----------
    filename : str
    path_col : str
        Name of the column taht contains path data.
    raw : bool, optional
        Unused, here for compatibility

    Returns
    -------
    pd.DataFrame

    """
    df = feather.read_feather(filename)
    series = df.get(path_col)

    if series is not None:
        for i, row in df.iterrows():
            arr = row[path_col]
            N = len(arr)
            df.at[i, path_col] = arr.reshape((N // 2, 2), order="C")

    return df


def previous_attempts(df: pd.DataFrame) -> pd.Series:
    """Extract the number of previous attempts from the metadata.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing path data and which has repeated attempts
        per user.

    Returns
    -------
    pd.Series
        The number of previous attempts.

    """
    assert "trajectory_data" in df, "error: DataFrame does not contain path data"
    out = df.trajectory_data.apply(lambda x: json.loads(x)["meta"]["previous_attempts"])
    out = out.rename("previous_attempts")

    return out


def duplicated_attempts(df: pd.DataFrame, keep: str = "first") -> pd.Series:
    """Compute the index of the last attempt of each player.

    For the computation, we first extract the number of previous attempts from
    the JSON data, we group by `user_id` and we take the min/max index for each
    group. NB: we use either of the functions `idx{min,max}` and therefore
    the correct row selection should make use of the function `loc`.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing path data and which has repeated attempts
        per user.
    keep : str, optional
        Valid options are 'first' (the default) and 'last' to decide which
        instance of the duplicated attempts to keep.

    Returns
    -------
    pd.Series
        The index of the unique attempts for each user.

    Example
    -------
    >>> idx = idx_last_attempt(df)
    >>> filtered_df = df.loc[idx]

    """
    assert "trajectory_data" in df, "error: DataFrame does not contain path data"
    assert keep in ("first", "last"), f"error: invalid option {keep}"

    # the series containing the number of previous attempts
    pa = previous_attempts(df)

    enlarged_df = pd.concat((df, pa), axis=1)
    gby = enlarged_df.groupby("user_id")["previous_attempts"]
    idx = gby.idxmin() if keep == "first" else gby.idxmax()

    return idx


def path(data: str) -> np.ndarray:
    """
    Parse JSON string and return an array representing a path.

    Parameters
    ----------
    data : str
        The input data.

    Returns
    -------
    np.array
        An Nx2 numpy array representing a path.

    """
    data = json.loads(data)["player"]

    if isinstance(data, dict):
        # The path is corrupted, the path is a single point
        # (and usually x is None)
        out = None
    else:
        out = map(lambda el: (el["x"], el["y"]), data)
        out = np.array(list(out))

    return out


def paths_from_df(
    df: pd.DataFrame, lexico: bool = False, grid_width: int = None
) -> Iterable[np.array]:
    """
    Parse JSON stored in a DataFrame and return a generator with paths.

    Parameters
    ----------
    df : pd.DataFrame
        The data loaded as a DataFrame.
    lexico : bool, optional
        Use lexicographic indexing.
    grid_width : int or None, optional
        Provide the width of the grid and use it to calculate the lexicographic
        index (default is None).

    Yields
    ------
    Iterable[np.array],
        A generator containing paths of size Nx2 (lexico False,
        default) or 2N (lexico=True; calculated as y * grid_width + x)

    """
    assert "trajectory_data" in df, "error: DataFrame does not contain path data"
    if lexico and grid_width is None:
        raise ValueError("grid_width is needed for lexicographic path")

    for i, row in df.iterrows():
        t = path(row.trajectory_data)

        if t is None:
            warnings.warn(f"corrupted data for entry: {i}")
            continue
        elif lexico:
            t = t[:, 1] * grid_width + t[:, 0]

        yield t


def _get_iterable(x):
    """Used to load either a single file or a list of files."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x
    else:
        return (x,)


def paths_from_files(
    files: Iterable[str], lexico: bool = False, grid_width: int = None
) -> Iterable[np.array]:
    """Parse paths from JSON files and return a generator.

    Parameters
    ----------
    files : List[str]
        The files to be loaded.
    lexico : bool, optional
        Use lexicographic indexing.
    grid_width : int or None, optional
        Provide the width of the grid and use it to calculate the lexicographic
        index (default is None).

    Yields
    ------
    Iterable[np.array],
        A generator containing paths of size Nx2 (lexico False,
        default) or 2N (lexico=True; calculated as y * grid_width + x)

    """
    if lexico and grid_width is None:
        raise ValueError("grid_width is needed for lexicographic path")

    for file in _get_iterable(files):
        with open(file, "r") as f:
            t = path(f.read())

        if t is None:
            continue
        elif lexico:
            t = t[:, 1] * grid_width + t[:, 0]

        yield t


def read_level_grid(
    filename: str, return_flags: bool = False
) -> Tuple[np.array, int, int]:
    """
    Load the boundaries and the dimension of the grid from the JSON file of a level.

    Parameters
    ----------
    filename : str
        The name of the json file.
    return_flags : bool, optional
        Return the coordinates of the checkpoints (defualt is False).

    Returns
    -------
    coords : np,.array
        The coordinates of the level landmass.
    width, length : int, int
        The width and length of the levels

    """
    data = _open_level_file(filename)

    wd, lg = data["grid_width"], data["grid_length"]
    grid = np.array(data["grid_data"]).reshape((wd, lg), order="F")
    coords = np.vstack(grid.nonzero()).T

    return coords, wd, lg


def read_level_size(filename: str) -> Tuple[int, int]:
    """
    Get the width and length from the JSON file of a level.

    Parameters
    ----------
    filename : str
        The name of the json file.

    Returns
    -------
    width, length : int, int

    """
    data = _open_level_file(filename)

    return data["grid_width"], data["grid_length"]


def read_level_flags(filename: str) -> Tuple[np.array, int, int]:
    """
    Get the coordinates of the flags (checkpoints) from the JSON file of a level.

    NB: the order in which the points are stored is not the order in which they
    should be visited, and the order in `flag_coords` cannot be guaranteed.

    Parameters
    ----------
    filename : str
        The name of the json file.

    Returns
    -------
    flag_coords : np.array

    """
    data = _open_level_file(filename)

    return np.array([(d["x"], d["y"]) for d in data["flags"]])


def _open_level_file(filename: str) -> Dict:
    """Load the data in a level json file."""

    if (ext := os.path.splitext(filename)[1]) != ".json":
        raise ValueError(f"unsupported format: {ext}")

    with open(filename, "r") as f:
        data = json.loads(f.read())
        assert "fixed" in data.keys(), "error: missing key `fixed`"
        data = data["fixed"]

    return data
