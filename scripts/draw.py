"""Different data and helper functions that are reused in different plots."""

import pickle

import matplotlib.pyplot as plt
from cycler import cycler

# The color cycler used for the ROC curves
set3_colors = list(plt.get_cmap("Set3").colors)
set3_colors.pop(1)  # the same as Peixoto does in graph-tool
set3_cycler = cycler(color=set3_colors)

cols = ["dur", "len", "curv", "bdy", "fro", "sup", "match", "mob"]

# titles
short = ["Dur.", "Len.", "Curv.", "Bdry.", "Frob", "Sup.", "Match.", "Mob."]
med = [
    "Dur.",
    "Len.",
    "Curv.",
    "Bdry. affty.",
    "Frob. dev.",
    "Sup. dev.",
    "Match. sum",
    "Mob. funct.",
]
long = [
    "Duration",
    "Length",
    "Total curvature",
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

# reverse_cols = ['len', 'curv', 'dtb', 'fro', 'inf']
# NB: more practical to reverse the remaining 2
reverse_cols = ["sum_match", "mob"]

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
    filename,
    col,
    levels=[6, 8, 11],
    legend=True,
    loc="lower right",
    verbose=False,
):
    if verbose:
        print("\nLoading: ", filename)

    with open(filename, "rb") as f:
        data = pickle.load(f)

    roc = data["roc_xy"]

    for lvl in levels:
        xy = roc[lvl][col]
        ax.plot(*xy, label=f"Level {lvl}", lw=2.5)

    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))
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
