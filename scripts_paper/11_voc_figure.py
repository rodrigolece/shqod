import os
from pathlib import Path
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

from draw import (
    cols,
    set3_cycler,
    plot_quartiles_by_vo,
    plot_roc_curves,
    plot_abs_correl,
)

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

# The default cycler
orig_cycler = plt.rcParams.get("axes.prop_cycle")


# Data
load_dir = Path("data_intermediate")
figures_dir = Path("figures")


def main(preffix):
    vo_filename = load_dir / f"{preffix}dataframe_level06.pkl"
    assert vo_filename.is_file()

    roc_filename = load_dir / f"{preffix}roc-auc_three-levels.pkl"
    assert roc_filename.is_file()

    with open(roc_filename, "rb") as f:
        roc_xy = pickle.load(f)["roc_xy"]

    # The correlations
    corr_filename = load_dir / f"{preffix}correls_three-levels.pkl"
    assert corr_filename.is_file()

    fig = plt.figure(figsize=(7.0, 10.0))  # contrained_layout=True

    gs_a = fig.add_gridspec(
        nrows=2,
        ncols=4,
        left=0.1,
        right=0.98,
        top=0.96,
        bottom=0.64,
        wspace=0.7,
        hspace=0.1,
    )
    gs_b = fig.add_gridspec(
        nrows=2, ncols=4, left=0.1, right=0.98, top=0.61, bottom=0.21, hspace=-0.3
    )
    gs_c = fig.add_gridspec(
        nrows=1,
        ncols=4,
        left=0.068,
        right=0.92,
        top=0.17,
        bottom=0.04,
        wspace=0.1,
        width_ratios=(1, 1, 1, 0.1),
    )

    panel_label_axes = []

    # Panel A
    for i, subplot_spec in enumerate(gs_a):
        ax = fig.add_subplot(subplot_spec)  # sharex=ax0 can be passed

        leg = i == 0
        plot_quartiles_by_vo(ax, vo_filename, cols[i], legend=leg)

        if i // 4 == 0:  # top row
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")

        if i == 0:
            panel_label_axes.append(ax)

    # Panel B
    plt.rcParams.update({"axes.prop_cycle": set3_cycler})

    for i, subplot_spec in enumerate(gs_b):
        ax = fig.add_subplot(subplot_spec, aspect="equal")

        leg = i == 0
        plot_roc_curves(ax, roc_xy, cols[i], legend=leg)

        if i // 4 == 0:  # top row
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")

        if i % 4 != 0:  # all but left col
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel("")

        if i == 0:
            panel_label_axes.append(ax)

    plt.rcParams.update({"axes.prop_cycle": orig_cycler})

    # Panel C

    axes_c = [fig.add_subplot(ss) for ss in gs_c]

    for i, lvl in enumerate([6, 8, 11]):
        ax = axes_c[i]

        if i < 3:
            ax.set_aspect("equal")

        cax = axes_c[-1] if i == 0 else None
        plot_abs_correl(
            ax, corr_filename, lvl, cax=cax
        )  # pad=0.02, shrink=0.74, aspect=12

        if i in (1, 2):
            plt.setp(ax.get_yticklabels(), visible=False)

        if i == 0:
            panel_label_axes.append(ax)

    # Background, labels, and annotations
    pu.Background(visible=False)
    pu.add_panel_labels(
        fig, axes=panel_label_axes, fontsize=18, xs=[-0.3, -0.2, -0.3]
    )  # ys

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()

    norm = args.norm
    preffix = "normed_" if norm else ""

    save = True

    fig = main(preffix)

    if save:
        filename = figures_dir / f"{preffix}panel_vo.pdf"  # svg
        fig.savefig(filename)  # bbox_inches="tight" mess the alignment
        print("Saved to: ", filename)

    print("\nDone!\n")
