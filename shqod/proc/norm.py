"""Compute the factor for normalization given a directory of features."""

from pathlib import Path

import pandas as pd

from shqod.io import LevelsLoader, write_feather
from shqod.paths import vo_correctness


def _gender_norm_factor(
    feat_loader, gender, ref_lvl=6, col="len", invert_cols=("match", "mob")
):
    # We use a reference df just to copy the index and filter out a chunk of the pop
    ref_ids = feat_loader.get(ref_lvl, gender)["id"]

    one = feat_loader.get(1, gender).set_index("id").reindex(ref_ids)
    two = feat_loader.get(2, gender).set_index("id").reindex(ref_ids)

    idx_one = vo_correctness(one.vo, 1)
    idx_two = vo_correctness(one.vo, 2)

    out = (one.loc[idx_one, col] + two.loc[idx_two, col]).abs()

    # Invert the features for which high in magnitude is good
    if col in invert_cols:
        out = 1 / out

    return out.reset_index()


def norm_factor(feat_loader, ref_lvl=6, col="len"):
    out = []

    for gender in ["f", "m"]:
        df = _gender_norm_factor(feat_loader, gender, ref_lvl=ref_lvl, col=col)
        df["gender"] = gender
        out.append(df)

    return pd.concat(out, axis=0, ignore_index=True)


def write_norm_factor(
    feat_dir,
    country="uk",
    fmt="feather",
    ref_lvl=6,
    col="len",
    verbose: bool = True,
):
    if fmt not in ("csv", "feather"):
        raise NotImplementedError

    feat_dir = Path(feat_dir)
    feat_loader = LevelsLoader(feat_dir, country=country, fmt="feather")

    if feat_loader._files:
        df = norm_factor(feat_loader, ref_lvl=ref_lvl, col=col)
        save_name = feat_dir / f"norm_{country}.{fmt}"

        if fmt == "feather":
            write_feather(df, save_name, verbose=verbose)

        elif fmt == "csv":
            df.to_csv(save_name)

    else:
        raise NameError("results could not be loaded")


def normalise_dataframe(
    feat_df,
    levels=(6, 8, 11),
    col="len",
    invert_cols=("match", "mob"),
    feat_types = ("dur", "len", "curv", "bdy", "fro", "sup", "match", "mob"),
    idx_cols=("id", "group"),
):
    feat_types = list(feat_types)
    idx_cols = list(idx_cols)

    gby = feat_df.groupby("level")

    # We compute the correction terms
    one = gby.get_group(1).set_index(idx_cols)
    two = gby.get_group(2).set_index(idx_cols)

    norm_factor = (one[col] + two[col]).abs()  # we keep only magnitude

    # Invert the features for which high in magnitude is good
    if col in invert_cols:
        norm_factor = 1 / norm_factor

    # Now we compute the normalised features
    out = []

    for lvl in levels:
        df = gby.get_group(lvl).set_index(idx_cols)
        df.loc[:, feat_types] = df[feat_types].divide(norm_factor, axis=0)
        out.append(df.reset_index())

    return pd.concat(out, ignore_index=True)
