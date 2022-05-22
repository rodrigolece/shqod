import os
from pathlib import Path
import pickle

import pandas as pd
from draw import cols


def main(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)["auc"]

    for gp in ["e3e4", "ad"]:
        df = data[gp]

        print("Group: ", gp)
        print(df.round(3))
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    suffix = "_normed" if args.norm else ""

    filename = input_dir / f"roc-auc_boxplots{suffix}.pkl"
    assert filename.is_file()

    main(filename)

    # Done
