import os
from pathlib import Path
import pickle

import pandas as pd
from sklearn import metrics

#  import pyarrow.feather as feather

from shqod import LevelsLoader, vo_correctness, compute_auc, compute_pvalues
from draw import cols, levels_shq


data_dir = Path(os.environ["dementia"]) / "data"
save_dir = Path("data_intermediate")


# The maps
grid_dir = data_dir / "maps"

# Normative
paths_dir = data_dir / "normative" / "paths"
features_dir = data_dir / "normative" / "features"

features_loader = LevelsLoader(features_dir)


def load_mixed_genders(level, age="50:", **kwargs):
    f = features_loader.get(level, "f", age=age, **kwargs)
    m = features_loader.get(level, "m", age=age, **kwargs)
    df = pd.concat((f, m)).reset_index(drop=True)
    idx = vo_correctness(df.vo, level, verbose=True)

    return df, idx


def roc_curves(norm, feat_types=cols):
    roc_xy_dict = {}

    for lvl in levels_shq:
        df, idx = load_mixed_genders(level=lvl, norm=norm, feat_types=feat_types)
        label = idx.astype(int).values

        for feat in feat_types:
            # roc definition has incorrect group (voc-0) first, hence minus sign
            score = -df[feat].values

            fpr, tpr, _ = metrics.roc_curve(label, score)  # 3rd argument is thresholds
            roc_xy_dict[(lvl, feat)] = (fpr, tpr)

            # auc = metrics.roc_auc_score(label, score)
            # auc_delong, variance = delong_roc_variance(label, score)
            # auc_dict[(lvl, feat)] = (auc, auc_delong, variance)

    return roc_xy_dict


def aucs(norm, feat_types=cols):
    out = []

    for lvl in levels_shq:
        df, idx = load_mixed_genders(level=lvl, norm=norm, feat_types=feat_types)
        idx = vo_correctness(df.vo, lvl)
        df["label"] = (~idx).astype(int)
        df = df.dropna()

        auc = compute_auc(df, feat_types).reset_index()
        auc["level"] = lvl
        out.append(auc)

    return pd.concat(out).set_index(["level", "metric"])


def pvalues(norm, feat_types=cols):
    out = []

    for lvl in levels_shq:
        df, idx = load_mixed_genders(level=lvl, norm=norm, feat_types=feat_types)
        idx = vo_correctness(df.vo, lvl)
        df["label"] = (~idx).astype(int)
        df = df.dropna()

        pvals = compute_pvalues(df, feat_types).reset_index()
        pvals["level"] = lvl
        out.append(pvals)

    return pd.concat(out).set_index(["level", "metric"])


def prepare_correls(norm, feat_types=cols):
    correls = {}

    for lvl in levels_shq:
        df, _ = load_mixed_genders(level=lvl, norm=norm, feat_types=feat_types)
        correls[lvl] = df[feat_types].corr()

    return correls


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()

    norm = args.norm
    suffix = "_normed" if norm else ""

    save = False

    # # Level 6 for the feature separation depending on VO
    # lvl = 6
    # df, idx = load_mixed_genders(level=lvl, norm=norm, feat_types=cols)
    # isna = df.len.isna()
    # df, idx = df.loc[~isna], idx[~isna]

    # filename = save_dir / f"dataframe_level{lvl:02}{suffix}.pkl"

    # if save:
    #     with open(filename, "wb") as f:
    #         pickle.dump({"df": df, "idx": idx}, f)
    #         print(f"\nSaved level {lvl} to: {filename}\n")

    # # ROC curves (three levels)
    # roc_xy = roc_curves(norm)
    # filename = save_dir / f"roc_three-levels{suffix}.pkl"

    # if save:
    #     with open(filename, "wb") as f:
    #         pickle.dump({"roc_xy": roc_xy}, f)
    #         print("\nSaved roc data to: ", filename)

    # AUC + p-vals (three levels)
    comparison_df = aucs(norm).join(pvalues(norm))
    filename = save_dir / f"auc-pvals_three-levels{suffix}.pkl"

    if True:
        comparison_df.to_pickle(filename)
        print("\nSaved auc data to: ", filename)

    # # Correlations (three levels)
    # correls = prepare_correls(norm)
    # filename = save_dir / f"correls_three-levels{suffix}.pkl"

    # if save:
    #     with open(filename, "wb") as f:
    #         pickle.dump({"correls": correls}, f)
    #         print("\nSaved correlation data to: ", filename)

    print("\nDone!\n")
