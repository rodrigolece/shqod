from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow.feather as feather


def write_feather(
    df: pd.DataFrame, filename: Union[str, Path], verbose: bool = True
) -> None:
    feather.write_feather(df, filename)

    if verbose:
        print("\nSaved results to:, ", filename)


def write_path_feather(
    df: pd.DataFrame, filename: Union[str, Path], verbose: bool = True
):
    pass
