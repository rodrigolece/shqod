import json
import pandas as pd


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
