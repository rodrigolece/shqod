import os
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats as st
import pyarrow.feather as feather

from shqod import LevelsLoader, compute_percentiles, compute_pvalues
from draw import cols, levels_shq


data_dir = Path(os.environ["dementia"]) / "data"
save_dir = Path("data_intermediate")


# The maps
# grid_dir = data_dir / "maps"

# Normative
features_dir = data_dir / "normative" / "features"
features_loader = LevelsLoader(features_dir, fmt="feather")

# Clinical
clinical_dir = data_dir / "clinical"
clinical_features_df = feather.read_feather(clinical_dir / "features.feather")
normed_features_df = feather.read_feather(clinical_dir / "normed_features.feather")


clinical_percentiles_df = compute_percentiles(
    clinical_features_df,
    features_loader,
    cols,
    filter_vo=True,
    fillna=np.inf,
)

normed_percentiles_df = compute_percentiles(
    normed_features_df,
    features_loader,
    cols,
    filter_vo=True,
    norm=True,
    fillna=np.inf,
)


def pvalues(df, other, ref="e3e3", feat_types=cols, equal_var=False):
    """Calculate p-values for the given groups."""

    groups = df.loc[df.group.isin([ref, other])].copy()
    groups["label"] = (groups["group"] == other).astype(int)

    gby = groups.groupby("level")

    out = []

    for lvl in levels_shq:
        lvl_df = gby.get_group(lvl)

        pvals = compute_pvalues(lvl_df, feat_types).reset_index()
        pvals["level"] = lvl
        out.append(pvals)

    return pd.concat(out).set_index(["level", "metric"])


if __name__ == "__main__":

    groups = ["e3e4", "ad"]

    save = False

    filenames = ["pvals_boxplots.csv", "pvals_boxplots_normed.csv"]
    dfs = [clinical_percentiles_df, normed_percentiles_df]

    headers = ["\nWithout normalisation", "\nCorrected for gameplay (levels 1 + 2)"]

    for data, fname, hdr in zip(dfs, filenames, headers):
        pvals = [pvalues(data, g) for g in groups]
        pvals_df = pd.concat(pvals, axis=1)
        pvals_df.columns = groups
        print(hdr)
        print(pvals_df.round(3))

        if save:
            filename = save_dir / fname
            pvals_df.to_csv(filename)
            print("\nSaved: ", filename)

    print("\nDone!\n")
