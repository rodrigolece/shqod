"""Compute the factor for normalization given a directory of features."""

from pathlib import Path
import pandas as pd

from shqod.io import LevelsLoader, write_feather
from shqod.paths import vo_correctness


def _gender_norm_factor(
    feat_loader, gender, ref_lvl=6, col="len", invert_cols=("match", "mob")
):
    # We use a reference df just to copy the index and filter out a chunk of the pop
    ref_ids = feat_loader.get(ref_lvl, g)["id"]

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
    feat_loader = LevelsLoader(feat_dir, country=country, fmt="feather")
    if fmt not in ("csv", "feather"):
        raise NotImplementedError

    if feat_loader._files:
        df = norm_factor(feat_loader, ref_lvl=ref_lvl, col=col)
        save_name = feat_dir / f"norm_{country}.{fmt}"

        if fmt == "feather":
            write_feather(df, save_name, verbose=verbose)

        elif fmt == "csv":
            df.to_csv(save_name)

    else:
        raise NameError("results could not be loaded")
