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
    parser.add_argument("filename", help="input data")
    args = parser.parse_args()

    # save_dir/roc-auc_boxplots.pkl
    filename = Path(args.filename)
    assert filename.is_file()

    main(filename)

    # Done
