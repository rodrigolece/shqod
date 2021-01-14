"""Read and write data."""

from typing import Tuple, Dict, Iterable, Union
# from .dtypes import Trajec, LexTrajec

import os
import warnings
import re
import json
import collections
import pyarrow.feather as feather
import numpy as np
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
        The loaded trajectory DataFrames, indexed by group keys (built from
        the names of the files by removing their extension).

    """

    def __init__(self, *dirs):
        """
        Load (a) directories(y) containing tidy CSV files.

        Parameters
        ----------
        *dirs : Variable length argument list
            The list of directories to load

        """
        self.loaded = dict()
        self._current_fmt = 'json'

        for d in dirs:
            self.loaded.update(self._load_csv_dir(d))

        return None

    def _load_csv_dir(self, directory: str) -> Dict[str, pd.DataFrame]:
        """Load csv files contained inside `directory`."""
        files = [os.path.join(directory, x)
                 for x in os.listdir(directory)]

        out = dict()

        for f in files:
            df = read_trajec_csv(f)
            key = os.path.basename(f).replace('.csv', '')
            out[key] = df

        return out

    def json_to_array(self) -> None:
        """Convert the `loaded` `trajectory_data` columns from json to array.

        Warning: this modifies the DataFrames in-place.

        """
        if self._current_fmt == 'json':
            self._current_fmt = 'array'

            for df in self.loaded.values():
                json_to_array(df, inplace=True)

        return None

    def get(self, level: int, gender: str = None) -> Dict[str, pd.DataFrame]:
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
        out = dict()

        for key, df in self.loaded.items():
            idx = (df.level == level)
            if gender:
                idx = idx & (df.gender == gender)

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
    def __init__(self, directory: str,
                 fmt: str = 'csv',
                 trajec: bool = True) -> Dict[str, str]:
        """
        Load the untidy CSV files contained within `directory`.

        Parameters
        ----------
        directory : str
        fmt : {'csv', 'feather'}
            Default is 'csv'.
        trajec : bool
            Whether the files being loaded should be treated as including
            trajectories (default is True).

        """
        assert fmt in ('csv', 'feather'), f'error: invalid format {fmt}'

        self.loaded = dict()
        self._files = dict()
        self._fmt = fmt
        self._trajec = trajec

        self._current_fmt = dict()  # used when the type is changed

        for file in os.listdir(directory):
            match = re.search(rf'_(\d+)_\w*_(f|m).{fmt}', file)
            if match:
                lvl, gender = match.groups()
                key = (int(lvl), gender)
                self._files[key] = os.path.join(directory, file)

        return None

    def json_to_array(self) -> None:
        """
        Convert the `loaded` `trajectory_data` columns from json to array.

        This function can only be called with data that was loaded using
        `trajec=True` as an argument, otherwise it raises an error.

        Warning: this modifies the DataFrames in-place.

        """
        if not self._trajec:
            raise ValueError('data was not loaded using trajec=True')

        for key, fmt in self._current_fmt.items():
            if fmt == 'json':
                self._current_fmt[key] = 'array'
                json_to_array(self.loaded[key], inplace=True)

        return None

    def get(self, level: int, gender: str, age: int = None) -> pd.DataFrame:
        """
        Get the DataFrame for a given level, gender and age.

        The CSV files are loaded as needed for a given level and gender. The
        resulting DataFrames are stored in the attribute `loaded` and the
        selection by age is a simple filtering operation.

        Parameters
        ----------
        level : int
        gender : {'f', 'm'}
        age : int, optional

        Returns
        -------
        pd.DataFrame

        """
        key = (level, gender)
        file = self._files.get(key)

        if file:
            if key not in self.loaded:
                # load trajectories file
                if self._trajec:
                    if self._fmt == 'csv':
                        method = read_trajec_csv
                        self._current_fmt[key] = 'json'
                    elif self._fmt == 'feather':
                        method = read_trajec_feather
                        self._current_fmt[key] = 'array'

                # load regular file
                else:
                    if self._fmt == 'csv':
                        method = pd.read_csv
                    elif self._fmt == 'feather':
                        method = feather.read_feather

                self.loaded[key] = method(file)

            df = self.loaded[key]

            # if age was passed as argument, filter
            out = df.loc[df.age == age] if age else df

        else:
            raise FileNotFoundError

        return out


def json_to_array(df: pd.DataFrame,
                  inplace: bool = False) -> Union[pd.DataFrame, None]:
    """Convert the `trajectory_data` column from json to array.

    Parameters
    ----------
    df : pd.DataFrame
    inplace : bool, optional
        If False, return a copy. Otherwise do operation in-place and
        return None

    """
    assert 'trajectory_data' in df,\
        'error: DataFrame does not contain trajectory data'

    if inplace:
        out = None

    else:
        df = df.copy()
        out = df

    for i, row in df.iterrows():
        t = trajec(row.trajectory_data)

        if t is None:
            warnings.warn('corrupted data; dropping row')
            df.drop(i, inplace=True)
        else:
            df.at[i, 'trajectory_data'] = t

    return out


def read_trajec_csv(filename: str) -> pd.DataFrame:
    """
    Read a csv file containing trajectory data into a DataFrame.

    Parameters
    ----------
    filename : str

    Returns
    -------
    pd.DataFrame

    """
    df = pd.read_csv(filename)
    assert 'trajectory_data' in df,\
        'error: file does not contain trajectory data'

    return df


def read_trajec_feather(filename: str) -> pd.DataFrame:
    """
    Read a feather file containing trajectory data into a DataFrame.

    Parameters
    ----------
    filename : str

    Returns
    -------
    pd.DataFrame

    """
    df = feather.read_feather(filename)
    assert 'trajectory_data' in df,\
        'error: file does not contain trajectory data'

    for i, row in df.iterrows():
        arr = row.trajectory_data
        N = len(arr)
        df.at[i, 'trajectory_data'] = arr.reshape((N//2, 2), order='C')

    return df


def previous_attempts(df: pd.DataFrame) -> pd.Series:
    """Extract the number of previous attempts from the metadata.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing trajectory data and which has repeated attempts
        per user.

    Returns
    -------
    pd.Series
        The number of previous attempts.

    """
    assert 'trajectory_data' in df,\
        'error: DataFrame does not contain trajectory data'
    out = df.trajectory_data.apply(
        lambda x: json.loads(x)['meta']['previous_attempts'])
    out = out.rename('previous_attempts')

    return out


def duplicated_attempts(df: pd.DataFrame, keep: str = 'first') -> pd.Series:
    """Compute the index of the last attempt of each player.

    For the computation, we first extract the number of previous attempts from
    the JSON data, we group by `user_id` and we take the min/max index for each
    group. NB: we use either of the functions `idx{min,max}` and therefore
    the correct row selection should make use of the function `loc`.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing trajectory data and which has repeated attempts
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
    assert 'trajectory_data' in df, \
        'error: DataFrame does not contain trajectory data'
    assert keep in ('first', 'last'), f'error: invalid option {keep}'

    # the series containing the number of previous attempts
    pa = previous_attempts(df)

    enlarged_df = pd.concat((df, pa), axis=1)
    gby = enlarged_df.groupby('user_id')['previous_attempts']
    idx = gby.idxmin() if keep == 'first' else gby.idxmax()

    return idx


def trajec(data: str) -> np.array:
    """
    Parse JSON string and return an array representing a trajectory.

    Parameters
    ----------
    data : str
        The input data.

    Returns
    -------
    np.array
        An Nx2 numpy array representing a trajectory.

    """
    data = json.loads(data)['player']

    if isinstance(data, dict):
        # The trajectory is corrupted, the trajectory is a single point
        # (and usually x is None)
        out = None
    else:
        out = map(lambda el: (el['x'], el['y']), data)
        out = np.array(list(out))

    return out


def trajecs_from_df(df: pd.DataFrame,
                    lexico: bool = False,
                    grid_width: int = None) -> Iterable[np.array]:
    """
    Parse JSON stored in a DataFrame and return a generator with trajectories.

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
        A generator containing trajectories of size Nx2 (lexico False,
        default) or 2N (lexico=True; calculated as y * grid_width + x)

    """
    assert 'trajectory_data' in df,\
        'error: DataFrame does not contain trajectory data'
    if lexico:
        assert grid_width is not None,\
            'error: grid_width is needed for lexicographic trajectory'

    for i, row in df.iterrows():
        t = trajec(row.trajectory_data)

        if t is None:
            warnings.warn(f'corrupted data for entry: {i}')
            continue
        elif lexico:
            t = t[:, 1] * grid_width + t[:, 0]

        yield t


def _get_iterable(x):
    """Used to load either a single file or a list of files."""
    if isinstance(x, collections.Iterable) and not isinstance(x, str):
        return x
    else:
        return (x,)


def trajecs_from_files(files: Iterable[str],
                       lexico: bool = False,
                       grid_width: int = None) -> Iterable[np.array]:
    """Parse trajectories from JSON files and return a generator.

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
        A generator containing trajectories of size Nx2 (lexico False,
        default) or 2N (lexico=True; calculated as y * grid_width + x)

    """
    if lexico:
        assert grid_width is not None,\
            'error: grid_width is needed for lexicographic trajectory'

    for file in _get_iterable(files):
        with open(file, 'r') as f:
            t = trajec(f.read())

        if t is None:
            continue
        elif lexico:
            t = t[:, 1] * grid_width + t[:, 0]

        yield t


def read_level_grid(filename: str,
                    return_flags: bool = False) -> Tuple[np.array, int, int]:
    """Short summary.

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
    flag_coords : np.array, optional
        The coordinates of the flags (checkpoints). NB: the order in which
        the points are stored is not the order in which they shoulb be
        visited, and the order in `flag_coords` cannot be used.

    """
    assert filename.endswith('.json'), 'error: invalid file type'

    with open(filename, 'r') as f:
        data = json.loads(f.read())
        assert 'fixed' in data.keys(), 'error: missing key `fixed`'
        data = data['fixed']

    wd, lg = data['grid_width'], data['grid_length']
    grid = np.array(data['grid_data']).reshape((wd, lg), order='F')
    coords = np.vstack(grid.nonzero()).T
    flag_coords = np.array([(d['x'], d['y']) for d in data['flags']])

    return (coords, wd, lg, flag_coords) if return_flags else (coords, wd, lg)
