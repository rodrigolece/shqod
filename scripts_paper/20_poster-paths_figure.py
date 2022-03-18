import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

import shqod
from draw import path, geometry
import plotutils as pu

# set non-interactive backend
mpl.use("agg")


# Set font sizes
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

# Data
data_dir = Path(os.environ["dementia"]) / "data"
clinical_paths = data_dir / "clinical" / "paths.feather"
grid_dir = data_dir / "maps"

# Output
figures_dir = Path("figures")


# Select the individual
ID = 5  # 5, 84, 97
group = "e3e4"

levels = [6, 8, 11]

paths_df = shqod.read_path_feather(clinical_paths, "trajectory_data")

df = paths_df.loc[
    (paths_df.group == group) & paths_df.level.isin(levels) & (paths_df.id == ID)
]


def main():
    screenshot = str(figures_dir / "screen.png")
    img = mpl.image.imread(screenshot)

    fig, axes = plt.subplots(
        1, 4, figsize=(7, 2.5), gridspec_kw={"width_ratios": [0.3, 0.25, 0.25, 0.25]}
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, wspace=0.5)

    ax = axes[0]
    ax.imshow(img)
    ax.set_axis_off()

    for i, ax in enumerate(axes[1:]):
        t = df.iloc[i].trajectory_data

        path(ax, t)
        geometry(ax, grid_dir, levels[i], fs=SMALL_SIZE)

        ax.set_aspect("equal")
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.axis("off")

    # Background, labels, and annotations
    bg = pu.Background(visible=False)
    pu.add_panel_labels(
        fig,
        axes=axes,
        fontsize=18,
        ys=[0, 1.01, 1.13, 1.0],
        labels=["", "B", "C", "D"],
        xs=0,
    )

    return fig


if __name__ == "__main__":

    save = True

    fig = main()

    if save:
        filename = figures_dir / "panel_screen-paths.pdf"
        fig.savefig(filename)  # bbox_inches="tight" mess the alignment
        print("Saved to: ", filename)

    print("\nDone!\n")
