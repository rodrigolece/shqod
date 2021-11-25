import warnings
import json

import numpy as np
import pandas as pd
import pyarrow.feather as feather


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


def jsoncol_to_array(path_col: pd.Series) -> pd.Series:
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
            df[path_col] = jsoncol_to_array(series)

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
