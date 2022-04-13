import os
from pathlib import Path
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn import metrics
import pyarrow.feather as feather

from shqod import LevelsLoader, compute_percentiles, compute_auc
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


def roc_curves(
    df: pd.DataFrame, other: str, ref: str = "e3e3", feat_types: List[str] = cols
):
    roc_xy_dict = {}

    for g in [ref, other]:
        assert g in df.group.unique()

    idx_group = df.group.isin([ref, other])

    for lvl in levels_shq:
        lvl_df = df.loc[(df.level == lvl) & idx_group]
        label = lvl_df.group == other

        for feat in feat_types:
            score = lvl_df[feat]

            fpr, tpr, _ = metrics.roc_curve(label, score)  # 3rd argument is thresholds
            roc_xy_dict[(lvl, feat)] = (fpr, tpr)

            # auc_dict[(lvl, feat)] = metrics.roc_auc_score(label, score)

    return roc_xy_dict


def aucs(df, other, ref="e3e3", feat_types=cols):
    groups = df.loc[df.group.isin([ref, other])].copy()
    groups["label"] = (groups["group"] == other).astype(int)

    gby = groups.groupby("level")

    out = []

    for lvl in levels_shq:
        lvl_df = gby.get_group(lvl).dropna(subset=feat_types)

        auc = compute_auc(lvl_df, feat_types).reset_index()
        auc["level"] = lvl
        out.append(auc)

    return pd.concat(out).set_index(["level", "metric"])


if __name__ == "__main__":

    save = False

    filenames = ["roc-auc_boxplots.pkl", "roc-auc_boxplots_normed.pkl"]
    dfs = [clinical_percentiles_df, normed_percentiles_df]

    for data, fname in zip(dfs, filenames):
        auc = {}
        roc_xy = {}

        for gp in ["e3e4", "ad"]:
            roc_xy[gp] = roc_curves(data, gp, ref="e3e3")
            auc[gp] = aucs(data, gp, ref="e3e3")

        if save:
            filename = save_dir / fname
            with open(filename, "wb") as f:
                pickle.dump({"auc": auc, "roc_xy": roc_xy}, f)
                print("\nSaved roc-auc data to: ", filename)

    print("\nDone!\n")
