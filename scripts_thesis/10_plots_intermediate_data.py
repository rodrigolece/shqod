import os
from pathlib import Path
import pickle

import pandas as pd
from sklearn import metrics

#  import pyarrow.feather as feather

from shqod import LevelsLoader, vo_correctness
from draw import cols, levels_shq


data_dir = Path(os.environ["dementia"]) / "data"
save_dir = Path("data_intermediate")


# The maps
grid_dir = data_dir / "maps"

# Normative
paths_dir = data_dir / "normative" / "paths"
features_dir = data_dir / "normative" / "features"

#  paths_loader = LevelsLoader(paths_dir, fmt="feather")
features_loader = LevelsLoader(features_dir, fmt="feather")


# Clinical
#  clinical_features = data_dir / "clinical" / "features.feather"
#  clinical_paths = data_dir / "clinical" / "paths.feather"

#  clinical_features_df = feather.read_feather(clinical_features)


def load(level, gender, age="50:"):
    df = features_loader.get(level, gender, age=age)
    idx = vo_correctness(df.vo, level, verbose=True)

    return df, idx


def prepare_roc_auc_data(df, idx, feat_types=cols):
    roc_xy_dict = {}
    auc_dict = {}

    label = idx.astype(int)

    for feat in feat_types:
        score = -df[feat]  # roc definition has incorrect group (voc-0) first hence -1

        fpr, tpr, _ = metrics.roc_curve(label, score)  # 3rd argument is thresholds
        roc_xy_dict[feat] = (fpr, tpr)
        auc_dict[feat] = metrics.roc_auc_score(label, score)

    return roc_xy_dict, auc_dict


if __name__ == "__main__":

    save = False

    # Level 6 for the feature separation depending on VO
    lvl = 6
    f_df, f_idx = load(level=lvl, gender="f")
    m_df, m_idx = load(level=lvl, gender="m")

    filename = save_dir / f"dataframes_level{lvl:02}.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump({"f_df": f_df, "f_idx": f_idx, "m_df": m_df, "m_idx": m_idx}, f)
            print(f"\nSaved level {lvl} to: {filename}\n")

    # The 3 levels for AUC and ROC
    auc = {}
    roc_xy = {}
    correls = {}

    # We use level 6 wich was previously loaded
    g = "f"
    correls[(lvl, g)] = f_df.corr()
    roc_xy[(lvl, g)], auc[(lvl, g)] = prepare_roc_auc_data(f_df, f_idx)

    g = "m"
    roc_xy[(lvl, g)], auc[(lvl, g)] = prepare_roc_auc_data(m_df, m_idx)
    correls[(lvl, g)] = m_df.corr()

    levels_shq.remove(lvl)

    for lvl in levels_shq:
        for g in ["f", "m"]:
            df, idx = load(level=lvl, gender=g)
            roc_xy[(lvl, g)], auc[(lvl, g)] = prepare_roc_auc_data(df, idx)
            correls[(lvl, g)] = df.corr()

    filename = save_dir / "roc-auc_three-levels.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump({"auc": auc, "roc_xy": roc_xy}, f)
            print("\nSaved roc-auc data to: ", filename)

    # Correlations
    filename = save_dir / "correls_three-levels.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump({"correls": correls}, f)
            print("\nSaved correlation data to: ", filename)

    print("\nDone!\n")
