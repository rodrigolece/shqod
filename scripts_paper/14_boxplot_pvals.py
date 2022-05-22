import os
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats as st
import pyarrow.feather as feather

from shqod import LevelsLoader, compute_percentiles, compute_pvalues
from draw import cols as feat_types


data_dir = Path(os.environ["dementia"]) / "data"

# Normative
features_dir = data_dir / "normative" / "features"
features_loader = LevelsLoader(features_dir, fmt="feather")

# Clinical
clinical_dir = data_dir / "clinical"

demo_cols = ["id", "group", "age", "gender", "level"]


def percentiles(norm, levels=(6, 8, 11)):
    preffix = "normed_" if norm else ""
    filename = clinical_dir / f"{preffix}features.feather"
    feat_df = feather.read_feather(filename)

    idx = feat_df.level.isin(levels)
    drop_cols = set(feat_df.columns).difference(feat_types + demo_cols)

    feat_df = feat_df.loc[idx].drop(columns=drop_cols)

    out = compute_percentiles(
        feat_df,
        features_loader,
        feat_types,
        filter_vo=True,
        norm=norm,
        fillna=np.inf,
    )

    return out


def pvalues(
    df, other, ref="e3e3", feat_types=feat_types, equal_var=False, levels=(6, 8, 11)
):
    """Calculate p-values for the given groups."""

    groups = df.loc[df.group.isin([ref, other])].copy()
    groups["label"] = (groups["group"] == other).astype(int)

    gby = groups.groupby("level")

    out = []

    for lvl in levels:
        lvl_df = gby.get_group(lvl)

        pvals = compute_pvalues(lvl_df, feat_types).reset_index()
        pvals["level"] = lvl
        out.append(pvals)

    return pd.concat(out).set_index(["level", "metric"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data_intermediate")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.outdir)

    groups = ["e3e4", "ad"]

    clinical_percentiles_df = percentiles(norm=True)
    normed_percentiles_df = percentiles(norm=False)
    dfs = [clinical_percentiles_df, normed_percentiles_df]

    filenames = ["pvals_boxplots.csv", "pvals_boxplots_normed.csv"]
    headers = ["\nWithout normalisation", "\nCorrected for gameplay (levels 1 + 2)"]

    for data, fname, hdr in zip(dfs, filenames, headers):
        pvals = [pvalues(data, g) for g in groups]
        pvals_df = pd.concat(pvals, axis=1)
        pvals_df.columns = groups
        print(hdr)
        print(pvals_df.round(3))

        if args.save:
            filename = save_dir / fname
            pvals_df.to_csv(filename)
            print("\nSaved: ", filename)

    print("\nDone!\n")
