import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import pyarrow.feather as feather


data_dir = Path(os.environ["dementia"]) / "data"


feat_types = ["dur", "len", "curv", "bdy", "fro", "sup", "match", "mob"]
invert_cols = ["match", "mob"]
idx_cols = ["id", "group"]


def normalise(feat_df, levels):
    gby = feat_df.groupby("level")

    # We compute the correction terms
    one = gby.get_group(1).set_index(idx_cols)
    two = gby.get_group(2).set_index(idx_cols)

    correction = (one[feat_types] + two[feat_types]).abs()  # we keep only magnitude

    # Invert the features for which high in magnitude is good
    for col in invert_cols:
        correction.loc[:, col] = 1 / correction[col]

    # Now we compute the normalised features
    out = []

    for lvl in tqdm(levels):
        df = gby.get_group(lvl).set_index(idx_cols)
        df.loc[:, feat_types] = df[feat_types] / correction[feat_types]
        out.append(df.reset_index())

    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":

    save = False

    clinical_features = data_dir / "clinical" / "features.feather"
    feat_df = feather.read_feather(clinical_features)
    levels = [6, 8, 11]

    norm_df = normalise(feat_df, levels)

    if save:
        save_name = clinical_features.parent / "normed_features.feather"
        feather.write_feather(norm_df, save_name, compression="zstd")
        print("\nSaved results to: ", save_name)
