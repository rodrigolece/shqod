import os
from pathlib import Path
import pickle

import pandas as pd
from sklearn import metrics

#  import pyarrow.feather as feather

from shqod import LevelsLoader, vo_correctness


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


keys = ["dur", "len", "curv", "bdy", "fro", "sup", "match", "mob"]
levels = [6, 8, 11]

# Used for the ROC curves
#  signs = dict(zip(keys, [-1] * 6 + [1] * 2))
signs = dict(zip(keys, [-1] * 8))


def load_mixed_genders(level, age="50:"):
    f = features_loader.get(level, "f", age=age)
    m = features_loader.get(level, "m", age=age)
    df = pd.concat((f, m)).reset_index(drop=True)
    idx = vo_correctness(df.vo, level, verbose=True)

    return df, idx


def prepare_roc_auc_data(df, idx, sign_dict=signs):
    roc_xy_dict = {}
    auc_dict = {}

    label = idx.astype(int)

    for feat, s in sign_dict.items():
        score = df[feat] * s

        fpr, tpr, _ = metrics.roc_curve(label, score)  # 3rd argument is thresholds
        roc_xy_dict[feat] = (fpr, tpr)
        auc_dict[feat] = metrics.roc_auc_score(label, score)

    return roc_xy_dict, auc_dict


if __name__ == "__main__":
    save = False

    # Level 6 for the feature separation depending on VO
    lvl = 6
    df, idx = load_mixed_genders(level=lvl)

    filename = save_dir / f"dataframe_level{lvl:02}.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump({"df": df, "idx": idx}, f)
            print(f"\nSaved level {lvl} to: {filename}\n")

    # The 3 levels for AUC and ROC
    auc = {}
    roc_xy = {}
    correls = {}

    # We use level 6 wich was previously loaded
    roc_xy[lvl], auc[lvl] = prepare_roc_auc_data(df, idx)
    correls[lvl] = df.corr()
    levels.remove(lvl)

    for lvl in levels:
        df, idx = load_mixed_genders(level=lvl)
        roc_xy[lvl], auc[lvl] = prepare_roc_auc_data(df, idx)
        correls[lvl] = df.corr()

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
