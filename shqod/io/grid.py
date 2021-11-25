import os
import json
from typing import Tuple, Dict

import numpy as np


def _open_level_file(filename: str) -> Dict:
    """Load the data in a level json file."""

    if (ext := os.path.splitext(filename)[1]) != ".json":
        raise ValueError(f"unsupported format: {ext}")

    with open(filename, "r") as f:
        data = json.loads(f.read())
        assert "fixed" in data.keys(), "error: missing key `fixed`"
        data = data["fixed"]

    return data


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
