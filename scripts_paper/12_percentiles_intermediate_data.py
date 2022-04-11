import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
import pyarrow.feather as feather

from shqod import LevelsLoader, compute_percentiles
from draw import cols, levels_shq


data_dir = Path(os.environ["dementia"]) / "data"
save_dir = Path("data_intermediate")


# Normative
features_dir = data_dir / "normative" / "features"
features_loader = LevelsLoader(features_dir, fmt="feather")


# Clinical
clinical_dir = data_dir / "clinical"
clinical_features_df = feather.read_feather(clinical_dir / "features.feather")
normed_features_df = feather.read_feather(clinical_dir / "normed_features.feather")

idx = clinical_features_df.level.isin(levels_shq)
demo_cols = ["id", "group", "age", "gender", "level"]
drop_cols = set(clinical_features_df.columns).difference(cols + demo_cols)

clinical_features_df = clinical_features_df.loc[idx].drop(columns=drop_cols)
normed_features_df = normed_features_df.drop(columns=drop_cols)


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


def make_data_long(df, level, feat_types=cols, idx_on=["id", "group"]):
    stubs = ["feat_" + x for x in feat_types]
    stub_dict = dict(zip(feat_types, stubs))
    lvl_df = df.loc[df.level == level]

    out = pd.wide_to_long(
        lvl_df.rename(columns=stub_dict),
        stubnames="feat",
        i=idx_on,
        j="type",
        sep="_",
        suffix=r"\w+",
    ).reset_index()

    return out


if __name__ == "__main__":

    save = True

    long_dict = {}
    norm_long_dict = {}

    for lvl in levels_shq:
        long_dict[lvl] = make_data_long(clinical_percentiles_df, lvl)
        norm_long_dict[lvl] = make_data_long(normed_percentiles_df, lvl)

    save_name = save_dir / "clinical-long-percentiles_three-levels.pkl"
    normed_sname = save_dir / "normed-clinical-long-percentiles_three-levels.pkl"

    if save:
        with open(save_name, "wb") as f:
            pickle.dump(long_dict, f)
            print("\nSaved long data to: ", save_name)

        with open(normed_sname, "wb") as f:
            pickle.dump(norm_long_dict, f)
            print("\nSaved long data to: ", normed_sname)

    print("\nDone!\n")
