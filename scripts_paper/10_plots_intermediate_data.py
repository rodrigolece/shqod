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


def load_mixed_genders(level, age="50:", **kwargs):
    f = features_loader.get(level, "f", age=age, **kwargs)
    m = features_loader.get(level, "m", age=age, **kwargs)
    df = pd.concat((f, m)).reset_index(drop=True)
    idx = vo_correctness(df.vo, level, verbose=True)

    return df, idx


def prepare_roc_auc_data(feat_types=cols):
    roc_xy_dict = {}
    auc_dict = {}

    for lvl in levels_shq:
        df, idx = load_mixed_genders(level=lvl)
        label = idx.astype(int)

        for feat in feat_types:
            # roc definition has incorrect group (voc-0) first, hence minus sign
            score = -df[feat]

            fpr, tpr, _ = metrics.roc_curve(label, score)  # 3rd argument is thresholds
            roc_xy_dict[(lvl, feat)] = (fpr, tpr)

            auc_dict[(lvl, feat)] = metrics.roc_auc_score(label, score)

    return roc_xy_dict, auc_dict


def prepare_correls():
    correls = {}

    for lvl in levels_shq:
        df, _ = load_mixed_genders(level=lvl)
        correls[lvl] = df.corr()

    return correls


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()

    norm = args.norm
    preffix = "normed_" if norm else ""

    features_loader = LevelsLoader(features_dir)

    save = True

    # Level 6 for the feature separation depending on VO
    lvl = 6
    df, idx = load_mixed_genders(level=lvl, norm=norm, feat_types=cols)
    isna = df.len.isna()
    df, idx = df.loc[~isna], idx[~isna]

    filename = save_dir / f"{preffix}dataframe_level{lvl:02}.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump({"df": df, "idx": idx}, f)
            print(f"\nSaved level {lvl} to: {filename}\n")

    # The 3 levels for AUC and ROC
    roc_xy, auc = prepare_roc_auc_data()
    filename = save_dir / f"{preffix}roc-auc_three-levels.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump({"auc": auc, "roc_xy": roc_xy}, f)
            print("\nSaved roc-auc data to: ", filename)

    # Correlations
    correls = prepare_correls()
    filename = save_dir / f"{preffix}correls_three-levels.pkl"

    if save:
        with open(filename, "wb") as f:
            pickle.dump({"correls": correls}, f)
            print("\nSaved correlation data to: ", filename)

    print("\nDone!\n")
