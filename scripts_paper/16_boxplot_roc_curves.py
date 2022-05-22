import os
from pathlib import Path
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

from draw import set3_cycler, plot_roc_curves
from draw import cols as feat_types
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


work_dir = Path(os.environ["dementia"]) / "code" / "shqod" / "scripts_paper"
figures_dir = work_dir / "figures"


def panels(filename, gp, feat_types=feat_types):
    assert gp in ("e3e4", "ad")

    with open(filename, "rb") as f:
        roc_xy = pickle.load(f)["roc_xy"]

    with mpl.rc_context({"axes.prop_cycle": set3_cycler}):
        fig = plt.figure(figsize=(7.0, 4.0))  # contrained_layout=True

        gs = fig.add_gridspec(
            nrows=2, ncols=4, left=0.1, right=0.98, top=1.00, bottom=0.02, hspace=-0.1
        )
        axes = gs.subplots(sharex=True, sharey=True).reshape(-1)

        for i, subplot_spec in enumerate(gs):
            ax = axes[i]

            leg = i == 0
            plot_roc_curves(ax, roc_xy[gp], feat_types[i], legend=leg)

            if i // 4 == 0:  # top row
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_xlabel("")

            if i % 4 != 0:  # all but left col
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_ylabel("")

        # Background, labels, and annotations
        pu.Background(visible=False)
        # pu.add_panel_labels(
        #     fig, axes=panel_label_axes, fontsize=18, xs=[-0.3, -0.2, -0.3]
        # )  # ys

        return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    figures_dir = Path(args.outdir)
    suffix = "_normed" if args.norm else ""

    filename = input_dir / f"roc-auc_boxplots{suffix}.pkl"
    assert filename.is_file()
    print("Loading data: ", filename)

    # The figures
    for gp in ["e3e4", "ad"]:
        fig = panels(filename, gp)
        save_name = figures_dir / f"roc-boxplots_{gp[-2:]}{suffix}.pdf"
        fig.savefig(save_name)
        print("Saved to: ", save_name)

    print("\nDone!\n")
