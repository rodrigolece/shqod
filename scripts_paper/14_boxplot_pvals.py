import os
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats as st
import pyarrow.feather as feather

from shqod import LevelsLoader, compute_percentiles
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

    idx1 = df.group == ref
    idx2 = df.group == other

    out = []

    for lvl in levels_shq:
        idx_lvl = df.level == lvl

        first = df.loc[idx_lvl & idx1]
        second = df.loc[idx_lvl & idx2]

        pvals = [
            st.ttest_ind(first[col], second[col], equal_var=equal_var).pvalue
            for col in feat_types
        ]
        row = pd.DataFrame([pvals], columns=cols)
        row["level"] = lvl

        out.append(row)

    return pd.concat(out, ignore_index=True)[["level"] + feat_types]


if __name__ == "__main__":

    save = False

    for gp in ["e3e4", "ad"]:
        print("\nGroup: ", gp)

        print("\nWithout normalisation")
        pvals = pvalues(clinical_percentiles_df, gp)
        print(pvals.round(3))

        if save:
            filename = save_dir / f"pvals-ttest-ind_boxplots_gp-{gp}.csv"
            pvals.to_csv(filename, index=False)
            print("\nSaved: ", filename)

        print("\nCorrected for gameplay (levels 1 + 2)")
        pvals = pvalues(normed_percentiles_df, gp)
        print(pvals.round(3))

        if save:
            filename = save_dir / f"normed-pvals-ttest-ind_boxplots_gp-{gp}.csv"
            pvals.to_csv(filename, index=False)
            print("\nSaved: ", filename)

    print("\nDone!\n")
