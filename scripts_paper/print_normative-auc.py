import pickle
import pandas as pd
import tabulate

with open("data_intermediate/roc-auc_three-levels.pkl", "rb") as f:
    data = pickle.load(f)["auc"]

levels = list(data.keys())

tab = []

for metric in data[levels[0]].keys():
    level = [metric]

    for lvl in levels:
        level.append(data[lvl][metric])

    tab.append(level)

df = pd.DataFrame(tab, columns=["metric"] + levels)
df["mean"] = df[levels].mean(axis=1)

tab = tabulate.tabulate(
    df.values.tolist(),
    headers=df.columns,
    floatfmt=".3f",
    tablefmt="latex",
)
print(tab)
