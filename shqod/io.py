"""Read and write data."""

from typing import List, Iterable
from .dtypes import Trajec, LexTrajec

import json


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
