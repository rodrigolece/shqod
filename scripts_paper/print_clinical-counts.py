import os
from pathlib import Path

import pandas as pd
import tabulate
import pyarrow.feather as feather

from shqod import vo_correctness
from draw import levels_shq


data_dir = Path(os.environ["dementia"]) / "data"

# Clinical
clinical_features = data_dir / "clinical" / "features.feather"
clinical_paths = data_dir / "clinical" / "paths.feather"


safe_mode = False

paths_tab = []
feat_tab = []

feat_df = feather.read_feather(clinical_features)
gby = feat_df.groupby("level")

tab = []

for lvl, group in gby:
    voc = vo_correctness(group.vo, lvl)
    lvl_df = pd.concat((group.group, voc), axis=1)
    lvl_df = lvl_df.groupby(["group", "vo"]).size().reset_index()
    lvl_df["level"] = lvl
    tab.append(lvl_df)

tab_df = pd.pivot_table(pd.concat(tab), columns="vo", index=["group", "level"])[0]
tab_df = tab_df.rename(columns={False: "Incorrect", True: "Correct"})
tab_df = tab_df.fillna(0).astype(int)

pctage = tab_df.copy()
pctage = pctage.div(pctage.sum(axis=1), axis=0) * 100

cols = ["Correct", "Correctp", "Incorrect", "Incorrectp"]
tab_df = tab_df.join(pctage, rsuffix="p")[cols].reset_index()

tab = tabulate.tabulate(
    tab_df.values.tolist(),
    headers=tab_df.columns,
    floatfmt=".1f",
    tablefmt="latex",
)
print(tab)

pivot_df = (
    pd.pivot_table(tab_df, values=cols, columns="level", index="group")
    .swaplevel(axis=1)[levels_shq]
    .reset_index()
)

# Now we get the columns ready for printing
pivot_df.columns = pivot_df.columns.droplevel(0)

tab = tabulate.tabulate(
    pivot_df.values.tolist(),
    headers=pivot_df.columns,
    floatfmt=".1f",
    tablefmt="latex",
)
print()
print(tab)

#  feat_tab.append([lvl, g, len(feat_df)])

#  if safe_mode:
#      paths_df = read_path_feather(clinical_paths, "trajectory_data")
#      paths_tab.append([lvl, g, len(paths_df)])

#  feat_df = pd.DataFrame(feat_tab, columns=["lvl", "g", "N"])

#  if safe_mode:
#      paths_df = pd.DataFrame(paths_tab, columns=["lvl", "g", "N"])
#      assert paths_df.N.sum() == feat_df.N.sum()

#  tab = tabulate.tabulate(feat_df.values.tolist(), headers=feat_df.columns)
#  print(tab)


#  tab = tabulate.tabulate(
#      pctage.values.tolist(),
#      headers=feat_df.columns,
#      floatfmt=".1f",
#  )

#  print("\nPercentage")
#  print(tab)
