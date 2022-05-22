import os
from pathlib import Path
import itertools
from multiprocessing import Pool

from tqdm import tqdm
import pyarrow.feather as feather

from shqod import LevelsLoader, AbsProcessor, RelProcessor, norm


data_dir = Path(os.environ["dementia"]) / "data"
grid_dir = data_dir / "maps"  # the maps

paths_dir = data_dir / "normative" / "paths"
output_dir = data_dir / "normative" / "features_modified"  # features_dir

paths_loader = LevelsLoader(paths_dir, fmt="feather")
norm_loader = paths_loader  # used inside the normative processor
# just a different name because it serves different purpose, but actually just
# the normative paths


# keys = ["len", "curv", "bdy"]
# nkeys = ["fro", "sup", "match", "mob", "vo"]
keys = ["curv", "bdy"]
nkeys = []


hp = {
    "grid_dir": grid_dir,
    "spline_res": 3,
    "window_size": 5,
    "weight_scale": 2.0,  # np.inf
    #  "max_sigdim": 4,
}

inner_bdy_radii = {
    1: {"bdy_rin": 1, "bdy_rout": 2},
    2: {"bdy_rin": 1, "bdy_rout": 2},
    6: {"bdy_rin": 1.5, "bdy_rout": 4},
    8: {"bdy_rin": 1.5, "bdy_rout": 4},
    11: {"bdy_rin": 1, "bdy_rout": 2},
}


levels = [1, 2, 6, 8, 11]
genders = ["f", "m"]
nb_iters = len(levels) * len(genders)


def process_level_gender(key):
    lvl, g = key
    df = paths_loader.get(lvl, g, age="24:80")

    lvl_hp = hp.copy()
    lvl_hp.update(inner_bdy_radii[lvl])

    proc = AbsProcessor(lvl, g, **lvl_hp)
    feat_df = proc.get_smooth_features(df, keys).rename(columns={"duration": "dur"})

    # norm_proc = RelProcessor(norm_loader, lvl, g, **lvl_hp)
    # nfeat_df = norm_proc.get_windowed_features(df, nkeys)

    # Combine the absolute and relative features
    # feat_df = feat_df.join(nfeat_df[nkeys])

    # Write file
    if True:
        filename = output_dir / f"level_{lvl}_uk_{g}.feather"
        feather.write_feather(feat_df, filename)

    # This should free up memory, though in paralell it might not help much
    paths_loader.loaded.pop((lvl, g), None)


if __name__ == "__main__":

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with Pool() as p:
        iterable = itertools.product(levels, genders)
        list(tqdm(p.imap(process_level_gender, iterable), total=nb_iters))
        # The outer list is required because the evaluation is lazy

    norm.write_norm_factor(output_dir)
