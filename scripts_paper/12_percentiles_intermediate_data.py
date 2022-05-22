import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
import pyarrow.feather as feather

from shqod import LevelsLoader, compute_percentiles
from draw import cols as feat_types


data_dir = Path(os.environ["dementia"]) / "data"

# Normative
features_dir = data_dir / "normative" / "features"
features_loader = LevelsLoader(features_dir, fmt="feather")

# Clinical
clinical_dir = data_dir / "clinical"

demo_cols = ["id", "group", "age", "gender", "level"]


def main(norm, levels=(6, 8, 11)):
    preffix = "normed_" if norm else ""
    filename = clinical_dir / f"{preffix}features.feather"
    feat_df = feather.read_feather(filename)
    # normed_features_df = feather.read_feather(filename)

    idx = feat_df.level.isin(levels)
    drop_cols = set(feat_df.columns).difference(feat_types + demo_cols)

    feat_df = feat_df.loc[idx].drop(columns=drop_cols)
    # normed_features_df = normed_features_df.drop(columns=drop_cols)

    p_df = compute_percentiles(
        feat_df,
        features_loader,
        feat_types,
        filter_vo=True,
        norm=norm,
        fillna=np.inf,
    )

    out = {}

    for lvl in levels:
        out[lvl] = make_data_long(p_df, lvl)

    return out


def make_data_long(df, level, feat_types=feat_types, idx_on=["id", "group"]):
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data_intermediate")
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.outdir)
    suffix = "_normed" if args.norm else ""

    long_dict = main(args.norm)
    filename = save_dir / f"clinical-long-percentiles_three-levels{suffix}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(long_dict, f)
        print("\nSaved to: ", filename)

    print("\nDone!\n")
