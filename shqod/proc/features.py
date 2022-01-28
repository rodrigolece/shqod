"""Some processing functions for features."""

import itertools
from typing import List

import numpy as np
import scipy.stats as st
import pandas as pd

from shqod.io import LevelsLoader
from shqod.utils import _get_iterable


def compute_percentiles(
    df: pd.DataFrame,
    loader: LevelsLoader,
    feat_types: List[str],
    filter_vo: bool = False,
    fillna: float = np.inf,
    drop_sig: bool = True,
) -> pd.DataFrame:
    """
    Compute the percentile score for a set of features and a reference pop.

    Parameters
    ----------
    df : pd.DataFrame
        The input data; it should contain the calculated features.
    loader : LevelsLoader
       The loader that get the DataFrames for the normative population; these
       should also contain calculated features.
    feat_types : List[str]
        The names of the features columns to use.
    filter_vo : bool, optional
        When set to True, we take as a reference the subset of the normative
        population that has the correct visiting order.
    fillna : np.float, optional
        Replace missing values with a given value (usually np.inf). None can
        also be used to disable the fill behaviour.
    drop_sig: bool, optional
        Whether to ignore path signature features (default is True). This
        option overrides any names that could be contained inside the list
        `feat_types`.

    Returns
    -------
    pd.DataFrame
        The output mimics the shape of the input `df` with the original entries
        replaced by numbers in [0, 100] that represent the percentiles.

    """
    if drop_sig:  # do not compute percentiles for path signature
        drop_cols = filter(lambda c: c.startswith("sig"), df.columns)
        out = df.drop(columns=drop_cols)
    else:
        out = df.copy()

    levels = df.level.unique()
    genders = df.gender.unique()

    if fillna:
        out.loc[:, feat_types] = out[feat_types].fillna(fillna)

    # percentile computation for each level and gender
    for lvl, g in itertools.product(levels, genders):
        idx = (out.level == lvl) & (out.gender == g)

        for i, row in out.loc[idx].iterrows():
            ref = loader.get(lvl, g, row.age, filter_vo=filter_vo, verbose=False)
            # for each age

            for col in feat_types:
                scores, val = ref[col], row[col]
                out.loc[i, col] = st.percentileofscore(scores, val, kind="weak")
                # weak corresponds to the CDF definition

    return out


def postprocess_group_features(group_df, feat_types):
    """
    Compute the percentile score for a set of features and a reference pop.

    Parameters
    ----------
    group_df : pd.DataFrame
        The input data; it should correspond to a single gorup (normally
        dementia patients).
    feat_types : List[str]
        The names of the features columns to use.

    Returns
    -------
    pd.DataFrame
        The output extends the input data and fills the missing attemps with NaNs.

    """
    gby = group_df.groupby("level")

    # Select the lowest level which we assume holds the most ids
    keys = list(gby.groups.keys())
    min_level = min(keys)
    keys.remove(min_level)

    # The reference (lowest level)
    base_df = gby.get_group(min_level).set_index("id")
    demo_cols = list(set(base_df.columns).difference(feat_types))
    demo_cols.remove("level")

    out = [base_df]

    for k in keys:
        lvl = gby.get_group(k).set_index("id")

        extended = base_df[demo_cols].join(lvl[["level"] + feat_types])
        extended.loc[extended.level.isna(), "level"] = k  # fix the level
        extended.level = extended.level.astype(int)

        out.append(extended)

    out_df = pd.concat(out).reset_index()[group_df.columns]

    return out_df.sort_values(["level", "id"]).reset_index(drop=True)


def fill_missing_attempts(df, feat_types, missing_group="ad"):
    """
    Compute the percentile score for a set of features and a reference pop.

    Parameters
    ----------
    df : pd.DataFrame
        The input data.
    feat_types : List[str]
        The names of the features columns to use.
    missing_group : str, optional
        The name of the group(s) with missing attempts that need to be filled.

    Returns
    -------
    pd.DataFrame
        The output extends the input data and fills the missing attemps with NaNs.

    """
    # TODO: use get_iterable to make missing_group accept more than one

    gby = df.groupby("group")
    process_df = postprocess_group_features(gby.get_group(missing_group), feat_types)

    out = [process_df]

    for key, group in gby:
        if key != missing_group:
            out.append(group)

    return pd.concat(out).reset_index(drop=True)
