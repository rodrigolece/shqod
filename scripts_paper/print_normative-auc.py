import os
from pathlib import Path
import argparse

import pandas as pd
import tabulate

from draw import cols

load_dir = Path("data_intermediate")

parser = argparse.ArgumentParser()
parser.add_argument("--norm", action="store_true")
parser.add_argument("--latex", action="store_true")
args = parser.parse_args()

norm = args.norm
suffix = "_normed" if norm else ""
tablefmt = "latex" if args.latex else None

filename = load_dir / f"auc-pvals_three-levels{suffix}.pkl"
df = pd.read_pickle(filename)
print(df.round(3))

# df = series.unstack(level=1)[cols]
# df["mean"] = df.mean(axis=1)
# print(df.T.round(3))

# tab = tabulate.tabulate(
#     df.values.tolist(),
#     headers=df.columns,
#     floatfmt=".3f",
#     tablefmt=tablefmt,
# )
# print(tab)
