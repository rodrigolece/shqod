"""Read and write data."""

from typing import Union, Optional, Tuple, List, Iterable
from .dtypes import Trajec, LexTrajec

import json
import numpy as np
import pandas as pd

LoadedTrajec = Union[Iterable[Trajec], Iterable[LexTrajec]]


def read_trajec_csv(filename: str,
                    return_length: bool = False) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Read a csv file containing trajectory data into a DataFrame.

    Parameters
    ----------
    filename : str
        The name of the csv file.
    return_length : bool, optional
        Return the number of entries loaded (default is False).

    Returns
    -------
    Tuple[pd.DataFrame, Optional[int]]
        The loaded DataFrame and optionally its length.

    """
    df = pd.read_csv(filename)
    assert 'trajectory_data' in df, 'error: file does not contain trajectory data'
    nb_entries = len(df)

    return (df, nb_entries) if return_length else df


def idx_last_attempt(df: pd.DataFrame) -> pd.Series:
    """Compute the index of the last attempt of each player.

    For the computation, we first extract the number of previous attempts from
    the JSON data, we group by `user_id` and we take the maximum index for each
    group. Note that we use the function `idxmax` and therefore the coorect way
    of filtering the original DataFrame is using the function `loc`.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing trajectory data and which has repeated attempts
        per user.

    Returns
    -------
    pd.Series
        The index of the last attempts for each unique `user_id`.

    Example
    -------
    >>> idx = idx_last_attempt(df)
    >>> filtered_df = df.loc[idx]

    """
    assert 'trajectory_data' in df, 'error: DataFrame does not contain trajectory data'

    # We compute a series containing the number of previous attempts
    series = df.trajectory_data.apply(
        lambda x: json.loads(x)['meta']['previous_attempts'])
    series = series.rename('previous_attempts')

    enlarged_df = pd.concat((df, series), axis=1)
    idx = enlarged_df.groupby('user_id')['previous_attempts'].idxmax()
    # NB: idxmax returns an index and the correct way of accessign the new df is using
    # df.loc[idx]

    return idx


def trajecs_from_df(df: pd.DataFrame,
                    lexico: bool = False,
                    grid_width: int = None) -> LoadedTrajec:
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
        index (defualt is None).

    Returns
    -------
    Union[Iterable[Trajec], Iterable[LexTrajec]]
        Depending on the value set for the flag `lexico`, return a generator
        with trajectories in the form of (x,y) coordinates or as a single index:
            i = grid_width * y + x

    """
    assert 'trajectory_data' in df,\
        'error: DataFrame does not contain trajectory data'
    if lexico:
        assert grid_width is not None,\
            'error: grid_width is needed for lexicographic trajectory'

    for _, row in df.iterrows():
        data = json.loads(row.trajectory_data)['player']
        if isinstance(data, dict):
            # The trajectory is corrupted, the trajectory is a single point
            # (and usually x is None)
            continue

        if lexico:
            yield map(lambda el: el['y'] * grid_width + el['x'], data)
        else:
            yield map(lambda el: (el['x'], el['y']), data)


def trajecs_from_files(files: Iterable[str],
                       lexico: bool = False,
                       grid_width: int = None) -> LoadedTrajec:
    """Parse trajectories from JSON files and return a generator.

    Parameters
    ----------
    files : List[str]
        The files to be loaded.
    lexico : bool, optional
        Use lexicographic indexing.
    grid_width : int or None, optional
        Provide the width of the grid and use it to calculate the lexicographic
        index (defualt is None).

    Returns
    -------
    Union[Iterable[Trajec], Iterable[LexTrajec]]
        Depending on the value set for the flag `lexico`, return a generator
        with trajectories in the form of (x,y) coordinates or as a single index:
            i = grid_width * y + x

    """
    if lexico:
        assert grid_width is not None,\
            'error: grid_width is needed for lexicographic trajectory'

    for file in files:
        with open(file, 'r') as f:
            data = json.loads(f.read())['player']
            if isinstance(data, dict):
                # The trajectory is corrupted, the trajectory is a single point
                # (and usually x is None)
                continue

            if lexico:
                yield map(lambda el: el['y'] * grid_width + el['x'], data)
            else:
                yield map(lambda el: (el['x'], el['y']), data)


def read_level_grid(filename: str) -> Tuple[np.array, np.array, int, int]:
    """Short summary.

    Parameters
    ----------
    filename : str
        The name of the json file.

    Returns
    -------
    Tuple[np.array, np.array, int, int]
        The x, y coordinates of the landmass (for plotting set as on) and the
        widht and length of the level.

    """
    assert filename.endswith('.json'), 'error: invalid file type'

    with open(filename, 'r') as f:
        data = json.loads(f.read())
        assert 'fixed' in data.keys(), 'error: missing key `fixed`'
        data = data['fixed']

    width, length = data['grid_width'], data['grid_length']
    grid = np.array(data['grid_data']).reshape((width, length), order='F')
    x, y = grid.nonzero()

    return x, y, width, length
