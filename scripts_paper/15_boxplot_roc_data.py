import os
from pathlib import Path
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn import metrics
import pyarrow.feather as feather

from shqod import LevelsLoader, compute_percentiles, compute_auc
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


def roc_curves(df, other, ref="e3e3", feat_types=feat_types, levels=(6, 8, 11)):
    roc_xy_dict = {}

    for g in [ref, other]:
        assert g in df.group.unique()

    idx_group = df.group.isin([ref, other])

    for lvl in levels:
        lvl_df = df.loc[(df.level == lvl) & idx_group]
        label = lvl_df.group == other

        for feat in feat_types:
            score = lvl_df[feat]

            fpr, tpr, _ = metrics.roc_curve(label, score)  # 3rd argument is thresholds
            roc_xy_dict[(lvl, feat)] = (fpr, tpr)

            # auc_dict[(lvl, feat)] = metrics.roc_auc_score(label, score)

    return roc_xy_dict


def aucs(df, other, ref="e3e3", feat_types=feat_types, levels=(6, 8, 11)):
    groups = df.loc[df.group.isin([ref, other])].copy()
    groups["label"] = (groups["group"] == other).astype(int)

    gby = groups.groupby("level")

    out = []

    for lvl in levels:
        lvl_df = gby.get_group(lvl).dropna(subset=feat_types)

        auc = compute_auc(lvl_df, feat_types).reset_index()
        auc["level"] = lvl
        out.append(auc)

    return pd.concat(out).set_index(["level", "metric"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data_intermediate")
    args = parser.parse_args()

    save_dir = Path(args.outdir)

    clinical_percentiles_df = percentiles(norm=True)
    normed_percentiles_df = percentiles(norm=False)
    dfs = [clinical_percentiles_df, normed_percentiles_df]

    filenames = ["roc-auc_boxplots.pkl", "roc-auc_boxplots_normed.pkl"]

    for data, fname in zip(dfs, filenames):
        auc = {}
        roc_xy = {}

        for gp in ["e3e4", "ad"]:
            roc_xy[gp] = roc_curves(data, gp, ref="e3e3")
            auc[gp] = aucs(data, gp, ref="e3e3")

        filename = save_dir / fname
        with open(filename, "wb") as f:
            pickle.dump({"auc": auc, "roc_xy": roc_xy}, f)
            print("\nSaved roc-auc data to: ", filename)

    print("\nDone!\n")
