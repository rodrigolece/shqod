import os
from pathlib import Path
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow.feather as feather

from shqod import (
    read_path_feather,
    UntidyLoader,
    TrajProcessor,
    NormativeProcessor,
)


data_dir = Path(os.environ["dementia"]) / "data"
grid_dir = data_dir / "maps"  # the maps

paths_dir = data_dir / "normative" / "paths"
paths_loader = UntidyLoader(paths_dir, fmt="feather")
norm_loader = paths_loader  # used inside the normative processor
# just a different name because it serves different purpose, but actually just
# the normative paths

clinical_paths = data_dir / "clinical" / "paths.feather"
clinical_features = data_dir / "clinical" / "features.feather"  # save name
clinical_paths_df = read_path_feather(clinical_paths, path_col="trajectory_data")


abs_cols = ["len", "curv", "bdy"]
rel_cols = ["fro", "sup", "match", "mob", "vo"]
idx_cols = ["id", "group"]


hp = {
    "grid_dir": grid_dir,
    "spline_res": 3,
    "window_size": 5,
    "weight_scale": 2.0,  # np.inf
    #  "max_sigdim": 4,
}

inner_bdy_radii = {
    6: {"bdy_rin": 1.5, "bdy_rout": 4},
    8: {"bdy_rin": 1.5, "bdy_rout": 4},
    11: {"bdy_rin": 1, "bdy_rout": 2},
}


levels = [6, 8, 11]
genders = ["f", "m"]
nb_iters = len(levels) * len(genders)


gby = clinical_paths_df.groupby(["level", "gender"])


def process_level_gender(key):
    lvl, g = key
    df = gby.get_group(key)

    lvl_hp = hp.copy()
    lvl_hp.update(inner_bdy_radii[lvl])

    proc = TrajProcessor(lvl, g, **lvl_hp)
    feat_df = proc.get_smooth_features(df, abs_cols, keys=idx_cols)
    feat_df = feat_df.rename(columns={"duration": "dur"}).set_index(idx_cols)

    norm_proc = NormativeProcessor(norm_loader, lvl, g, **lvl_hp)
    nfeat_df = norm_proc.get_windowed_features(df, rel_cols, keys=idx_cols)
    nfeat_df = nfeat_df.set_index(idx_cols)

    # Combine the absolute and relative features
    feat_df = feat_df.join(nfeat_df[rel_cols]).reset_index()

    # This should free up memory
    paths_loader.loaded.pop((lvl, g), None)

    return feat_df


if __name__ == "__main__":
    with Pool() as p:
        iterable = itertools.product(levels, genders)
        features_df = pd.concat(
            tqdm(p.imap(process_level_gender, iterable), total=nb_iters)
        )

    # Write file
    feather.write_feather(features_df, clinical_features)