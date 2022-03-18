import os
from pathlib import Path

import pandas as pd
import tabulate

from shqod import LevelsLoader, vo_correctness
from draw import levels_shq


data_dir = Path(os.environ["dementia"]) / "data"

# Normative
paths_dir = data_dir / "normative" / "paths"
features_dir = data_dir / "normative" / "features"

paths_loader = LevelsLoader(paths_dir, fmt="feather")
features_loader = LevelsLoader(features_dir, fmt="feather")

# Clinical
#  clinical_features = data_dir / "clinical" / "features.feather"
#  clinical_paths = data_dir / "clinical" / "paths.feather"


safe_mode = False

counts_tab = []
vo_tab = []
path_counts_tab = []

for lvl in levels_shq:
    for g in ["f", "m"]:

        feat_df = features_loader.get(lvl, g, age="50:80")
        counts_tab.append([lvl, g, len(feat_df)])

        idx_vo = vo_correctness(feat_df.vo, lvl, verbose=False)
        vo_tab.append([lvl, g, idx_vo.sum()])

        if safe_mode:
            paths_df = paths_loader.get(lvl, g, age="50:80")
            path_counts_tab.append([lvl, g, len(paths_df)])

counts_df = pd.DataFrame(counts_tab, columns=["lvl", "g", "N"])
vo_df = pd.DataFrame(vo_tab, columns=["lvl", "g", "voc"])

if safe_mode:
    paths_df = pd.DataFrame(path_counts_tab, columns=["lvl", "g", "N"])
    assert paths_df.N.sum() == counts_df.N.sum()


# Print the counts for the two genders and the percentage each represents
pctage = counts_df.groupby(["lvl", "g"]).sum()["N"]
pctage = pctage.groupby(level=0).apply(lambda x: 100 * x / x.sum())  # .reset_index()
pctage.name = "%"
joined_df = counts_df.set_index(["lvl", "g"]).join(pctage).reset_index()

tab = tabulate.tabulate(
    joined_df.values.tolist(),
    headers=joined_df.columns,
    floatfmt=".1f",
    tablefmt="latex",
)

print(tab)


# Print the number and fraction of succesful VOs
counts_df = counts_df.set_index(["lvl", "g"])
vo_df = vo_df.set_index(["lvl", "g"])
vo_df["frac"] = 100 * vo_df.voc / counts_df.N
vo_df = vo_df.reset_index()

tab = tabulate.tabulate(
    vo_df.values.tolist(),
    headers=vo_df.columns,
    floatfmt=".1f",
    tablefmt="latex",
)

print("\n")
print(tab)
