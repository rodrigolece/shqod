"""Read and write data."""

from typing import Union, Optional, Tuple, List, Iterable
from .dtypes import Trajec, LexTrajec

import json
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
        The loaded datafram and optionally its length.

    """
    df = pd.read_csv(filename)
    assert 'trajectory_data' in df, 'error: file does not contain trajectory data'
    nb_entries = len(df)

    return (df, nb_entries) if return_length else df


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
        with trajectories in the form of (x,y) coordinates or as a single index.

    """
    assert 'trajectory_data' in df,\
        'error: DataFrame does not contain trajectory data'
    if lexico:
        assert grid_width is not None,\
            'error: grid_width is needed for lexicographic trajectory'

    for _, row in df.iterrows():
        data = json.loads(row.trajectory_data)['player']

        if lexico:
            yield map(lambda el: el['x'] * grid_width + el['y'], data)
        else:
            yield map(lambda el: (el['x'], el['y']), data)


def load_trajecs(files: Iterable[str]) -> List[Trajec]:
    """Load trajectory JSON data as a generator containing coordinates of points.

    Parameters
    ----------
    files : List[str]
        The files to be loaded.

    Returns
    -------
    List[Trajec]
        It contains a list of trajectories as a series of (x,y) points.

    """
    out = []

    for file in files:
        with open(file, 'r') as f:
            data = json.loads(f.read())
            out.append(map(lambda el: (el['x'], el['y']), data['player']))

    return out


def load_trajecs_lex(files: List[str],
                     grid_width: int) -> Iterable[LexTrajec]:
    """Load trajectory JSON data as a generator containing lexicographic points.

    Parameters
    ----------
    files : List[str]
        The files to be loaded.
    grid_width : int
        The width of the grid in the level.

    Returns
    -------
    Iterable[LexTrajec]
        It contains a list of trajectories as a series of lexicographic indices,
        where the coordinate index is given by
            i = grid_width * y + x

    """
    out = []

    for file in files:
        with open(file, 'r') as f:
            data = json.loads(f.read())
            trajec = map(lambda el: el['y'] * grid_width + el['x'],
                         data['player'])
            out.append(trajec)

    return out
