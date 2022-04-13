"""Some processing functions for features."""

import itertools
from typing import List, Union

import numpy as np
import scipy.stats as st
import pandas as pd

# from sklearn import metrics

from shqod.io import LevelsLoader
from shqod.utils import _get_iterable, confidence_interval
from shqod.auc import delong_roc_variance


def compute_percentiles(
    df: pd.DataFrame,
    loader: LevelsLoader,
    feat_types: List[str],
    filter_vo: bool = False,
    norm: bool = False,
    fillna: float = np.inf,
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
        The names of the feature columns to use.
    filter_vo : bool, optional
        When set to True, we take as a reference the subset of the normative
        population that has the correct visiting order.
    fillna : np.float, optional
        Replace missing values with a given value (usually np.inf). None can
        also be used to disable the fill behaviour.

    Returns
    -------
    pd.DataFrame
        The output mimics the shape of the input `df` with the original entries
        replaced by numbers in [0, 100] that represent the percentiles.

    """
    out = df.copy()

    levels = df.level.unique()
    genders = df.gender.unique()

    if fillna:
        out.loc[:, feat_types] = out[feat_types].fillna(fillna)

    # percentile computation for each level and gender
    for lvl, g in itertools.product(levels, genders):
        idx = (out.level == lvl) & (out.gender == g)

        for i, row in out.loc[idx].iterrows():
            ref = loader.get(
                lvl,
                g,
                row.age,
                filter_vo=filter_vo,
                norm=norm,
                feat_types=feat_types,
                verbose=False,
            )

            for col in feat_types:
                scores, val = ref[col], row[col]
                out.loc[i, col] = st.percentileofscore(scores, val, kind="weak")
                # weak corresponds to the CDF definition

    return out


def _fill_group(group_df, feat_types, ref_lvl=None):
    """
    TODO

    Parameters
    ----------
    group_df : pd.DataFrame
        The input data; it should correspond to a single group (normally
        dementia patients).
    feat_types : List[str]
        The names of the feature columns to use.

    Returns
    -------
    pd.DataFrame
        The output extends the input data and fills the missing attemps with NaNs.

    """
    gby = group_df.groupby("level")
    keys = list(gby.groups.keys())

    if ref_lvl is None:
        # Select the lowest level which we assume holds the most ids
        ref_lvl = min(keys)

    # The reference data
    base_df = gby.get_group(ref_lvl).set_index("id")
    demo_cols = list(set(base_df.columns).difference(feat_types))
    demo_cols.remove("level")

    out = [base_df]

    keys.remove(ref_lvl)
    for k in keys:
        lvl = gby.get_group(k).set_index("id")

        extended = base_df[demo_cols].join(lvl[["level"] + feat_types])
        extended.loc[extended.level.isna(), "level"] = k  # fix the level
        extended.level = extended.level.astype(int)

        out.append(extended)

    out_df = pd.concat(out).reset_index()[group_df.columns]

    return out_df.sort_values(["level", "id"]).reset_index(drop=True)


def fill_missing_attempts(df, feat_types, missing_group="ad", ref_lvl=None):
    """
    TODO

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
    group_df = gby.get_group(missing_group)

    process_df = _fill_group(group_df, feat_types, ref_lvl=ref_lvl)

    out = [process_df]

    for key, group in gby:
        if key != missing_group:
            out.append(group)

    return pd.concat(out).reset_index(drop=True)


def compute_pvalues(
    lvl_df: pd.DataFrame,
    feat_types: List[str],
    label: str = None,
    equal_var: bool = False,
):
    label = "label" if label is None else label
    if label not in lvl_df:
        raise ValueError

    label = lvl_df[label]
    gs = label.unique()
    assert len(gs) == 2

    idx = label == gs[0]
    first = lvl_df.loc[idx]
    second = lvl_df.loc[~idx]

    pvals = [
        st.ttest_ind(first[feat], second[feat], equal_var=equal_var).pvalue
        for feat in feat_types
    ]

    idx = pd.Index(feat_types, name="metric")

    return pd.Series(pvals, index=idx, name="pvals")


def _ci_wrapper(row, alpha=0.95):
    return confidence_interval(row["auc"], row["std"], alpha=alpha)


def compute_auc(lvl_df: pd.DataFrame, feat_types: List[str], label: str = None):
    label = "label" if label is None else label
    if label not in lvl_df:
        raise ValueError

    label = lvl_df[label].values  # delong_roc_var takes np.array
    vals = []

    for feat in feat_types:
        score = lvl_df[feat].values
        # auc = metrics.roc_auc_score(label, score)
        vals.append(delong_roc_variance(label, score))

    idx = pd.Index(feat_types, name="metric")
    out = pd.DataFrame(vals, columns=["auc", "var"], index=idx)

    out["std"] = out["var"].apply(np.sqrt)
    out[["CI_low", "CI_high"]] = out.apply(_ci_wrapper, axis=1, result_type="expand")

    return out.drop(columns="var")
