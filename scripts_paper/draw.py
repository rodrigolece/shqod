"""Different data and helper functions that are reused in different plots."""

import pickle

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns

import shqod

# The color cycler used for the ROC curves
set3_colors = list(plt.get_cmap("Set3").colors)
set3_colors.pop(1)  # the same as Peixoto does in graph-tool
set3_cycler = cycler(color=set3_colors)

cols = ["dur", "len", "curv", "bdy", "fro", "sup", "match", "mob"]

# titles
short = ["Dur.", "Len.", "Curv.", "Bdry.", "Frob.", "Sup.", "Match.", "Mob."]
med = [
    "Dur.",
    "Len.",
    "Curv.",
    "Bdry. affty.",
    "Frob.",
    "Sup.",
    "Match.",
    "Mob. funct.",
]
long = [
    "Duration",
    "Length",
    "Avg. curvature",
    "Boundary affinity",
    "Frobenius dev.",
    "Supremum dev.",
    "Matching-sum",
    "Mobility functional",
]

title = dict(zip(cols, long))
title_short = dict(zip(cols, med))
title_sshort = dict(zip(cols, short))

features = title.keys()
feat_types = cols

genders = ["f", "m"]
levels_shq = [6, 8, 11]

target_orders = {
    6: [0, 1, 2],
    8: [0, 1, 2],
    11: [1, 0, 1, 2],
}


def plot_quartiles(ax, df, col, label=None):
    data = df.groupby("age")[col].describe()
    ln = ax.plot(data["50%"], "-", label=label)
    # TODO: this is where I need to fix the legend so that it includes shaded areas
    poly = ax.fill_between(data.index, data["25%"], data["75%"], alpha=0.1)

    ax.set_xlabel("Age")
    ax.set_ylabel(title[col])

    return ln[0], poly


def plot_quartiles_by_vo(
    ax,
    filename,
    col,
    legend=True,
    loc="upper left",
    verbose=False,
):
    if verbose:
        print(f"Loading: ", filename)

    with open(filename, "rb") as f:
        data = pickle.load(f)

    df, vo_idx = data["df"], data["idx"]
    correct, incorrect = df.loc[vo_idx], df.loc[~vo_idx]

    art1 = plot_quartiles(ax, correct, col)
    art2 = plot_quartiles(ax, incorrect, col)

    if legend:
        handles = [art1, art2]
        labels = ["VO-C", "VO-I"]
        ax.legend(handles, labels, loc=loc)


def plot_roc_curves(
    ax,
    roc_xy,
    col,
    levels=[6, 8, 11],
    legend=True,
    loc="lower right",
    verbose=False,
):

    for lvl in levels:
        # xy = roc_xy[lvl][col]
        xy = roc_xy[(lvl, col)]
        ax.plot(*xy, label=f"Level {lvl}", lw=2.5)

    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))
    ax.set_aspect("equal")
    x, X, y, Y = ax.axis()
    mM = max(x, y), min(X, Y)
    ax.plot(mM, mM, ls="dotted", c=".3", alpha=0.5)  # , label="y=x (diagonal)")

    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title(title[col])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    if legend:
        ax.legend(loc=loc)  # fontsize=18


def plot_abs_correl(
    ax,
    filename,
    level,
    cax=None,
    verbose=False,
    **cbar_args,
):
    if verbose:
        print("\nLoading: ", filename)

    with open(filename, "rb") as f:
        data = pickle.load(f)

    level_correls = data["correls"][level]
    mat = level_correls.loc[cols, cols]  # to avoid including age

    mappable = ax.matshow(mat.abs(), vmin=0, vmax=1)

    ax.set_title(f"Level {level}", y=-0.22)
    ax.tick_params(bottom=False)

    ax.set_xticks(range(8))
    ax.set_xticklabels(title_sshort.values(), rotation=90, ha="left")

    ax.set_yticks(range(8))
    ax.set_yticklabels(title_sshort.values())

    if cax:
        plt.colorbar(mappable, cax=cax, **cbar_args)

    return mat


def boxplot(
    ax,
    filename,
    level,
    groups=["e3e3", "e3e4", "ad"],
    legend=True,
    loc="upper right",
    verbose=False,
):
    if verbose:
        print("\nLoading: ", filename)

    with open(filename, "rb") as f:
        data = pickle.load(f)

    data_df = data[level]
    data_df = data_df.loc[data_df.group.isin(groups)].replace(title_short)

    sns.boxplot(
        data=data_df,
        x="type",
        y="feat",
        hue="group",
        hue_order=groups,
        showcaps=False,
        showfliers=True,
        ax=ax,
    )

    Y = 105

    if legend:
        ax.legend(*ax.get_legend_handles_labels(), loc=loc, ncol=len(groups))
        Y += 20
    else:
        ax.legend([], [], frameon=False)

    ax.set_ylim((-5, Y))

    ax.set_xlabel("")
    ax.set_ylabel("Percentile")
    ax.set_title(f"Level {level}")


def path(ax, path, ms=40):
    start = path[0]
    spath = shqod.smooth(path)

    ax.scatter(path[:, 0], path[:, 1], s=10, c="0.6", zorder=1)
    ax.plot(spath[:, 0], spath[:, 1], "0.2", zorder=2)
    ax.scatter(*start, s=ms, c="C1", marker="^", zorder=3)


def geometry(ax, grid_dir, lvl, ms=50, fs=10, simple=True):
    map_filename = grid_dir / f"level{lvl:02}.json"
    flags = shqod.read_level_flags(map_filename)[::-1]

    if simple:
        inner_filename = grid_dir / f"inner_bdy_level{lvl:02}.npy"
        bdy = np.load(inner_filename)

        # We do a little shift and extension to hide the discontinuity in the
        # smooth interpolation of the map
        shift, ext = -5, 3

        bdy = np.roll(bdy, shift, axis=0)
        bdy = np.append(bdy, bdy[:ext], axis=0)
        sbdy = shqod.smooth(bdy)

        ax.plot(sbdy[:, 0], sbdy[:, 1], "gainsboro", zorder=0)

    else:
        land, _, _ = shqod.read_level_grid(map_filename)
        ax.scatter(land[:, 0], land[:, 1], c="gainsboro")

    ax.scatter(flags[:, 0], flags[:, 1], s=ms, c="C1", marker="s", zorder=4)
    for i, (x, y) in enumerate(flags):
        ax.text(x, y - 0.5, str(i + 1), ha="center", va="center", fontsize=fs, zorder=5)
