import os
from pathlib import Path
import pickle

import pandas as pd
import tabulate


# Data
load_dir = Path("data_intermediate")
figures_dir = Path("figures")


roc_filename = load_dir / "roc-auc_three-levels.pkl"
assert os.path.isfile(roc_filename)

# The correlations
corr_filename = load_dir / "correls_three-levels.pkl"
assert os.path.isfile(corr_filename)


with open(roc_filename, "rb") as f:
    data = pickle.load(f)
    auc = data["auc"]

keys = list(auc.keys())

tab = []

for metric in auc[keys[0]].keys():
    row = [metric]

    for k in keys:
        row.append(auc[k][metric])

    tab.append(row)

df = pd.DataFrame(tab, columns=["metric"] + keys)
df["mean"] = df[keys].mean(axis=1)

tab = tabulate.tabulate(
    df.values.tolist(),
    headers=df.columns,
    floatfmt=".3f",
    tablefmt="latex",
)
print(tab)
