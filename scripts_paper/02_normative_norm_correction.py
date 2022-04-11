import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow.feather as feather

from shqod import LevelsLoader, vo_correctness


data_dir = Path(os.environ["dementia"]) / "data"
# grid_dir = data_dir / "maps"  # the maps

feat_dir = data_dir / "normative" / "features"
feat_loader = LevelsLoader(feat_dir, fmt="feather")


feat_types = ["dur", "len", "curv", "bdy", "fro", "sup", "match", "mob"]
invert_cols = ["match", "mob"]
idx_cols = ["id", "group"]


def norm_sum(ref_lvl, g):
    # We use a reference df just to copy the index and filter out a chunk of the pop
    ref_df = feat_loader.get(ref_lvl, g).set_index("id").drop(columns="vo")

    one = feat_loader.get(1, g).set_index("id").reindex(ref_df.index)
    two = feat_loader.get(2, g).set_index("id").reindex(ref_df.index)

    idx_one = vo_correctness(one.vo, 1)
    idx_two = vo_correctness(one.vo, 2)

    ref_df.loc[:, feat_types] = (one[feat_types] + two[feat_types]).abs()

    # Invert the features for which high in magnitude is good
    for col in invert_cols:
        ref_df.loc[:, col] = 1 / ref_df[col]

    return ref_df.loc[idx_one & idx_two].reset_index()


if __name__ == "__main__":

    save = False

    ref_lvl = 6
    genders = ["f", "m"]

    for g in genders:
        df = norm_sum(ref_lvl, g)

        if save:
            save_name = feat_dir / f"norm_uk_{g}.feather"
            feather.write_feather(df, save_name, compression="zstd")
            print("\nSaved results to:, ", save_name)
