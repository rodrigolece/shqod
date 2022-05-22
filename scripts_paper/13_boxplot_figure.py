import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from draw import boxplot
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

# We select particular colors for the clinical groups
color_map = dict(zip(["e3e3", "e3e4", "e4e4", "ad"], orig_cycler.by_key()["color"][:4]))

work_dir = Path(os.environ["dementia"]) / "code" / "shqod" / "scripts_paper"
figures_dir = work_dir / "figures"


def main(boxplot_filename, groups, levels=(6, 8, 11)):
    fig = plt.figure(figsize=(7.0, 8.0))

    gs = fig.add_gridspec(
        3,
        1,
        left=0.1,
        right=0.98,
        bottom=0.04,
        top=0.96,
        hspace=0.3,
    )

    for i, subplot_spec in enumerate(gs):
        ax = fig.add_subplot(subplot_spec)

        leg = i == 0
        boxplot(ax, boxplot_filename, levels[i], legend=leg, groups=groups)

    pu.Background(visible=False)

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

    filename = input_dir / f"clinical-long-percentiles_three-levels{suffix}.pkl"
    assert filename.is_file()
    print("Loading data: ", filename)

    groups = ["e3e3", "e3e4", "ad"]
    # custom_cycler = cycler(color=[color_map[g] for g in groups])
    custom_cycler = cycler(color=["#00ABE8", "#FF7F0E", "#EB1928"])  # Uzu's edit
    plt.rcParams.update({"axes.prop_cycle": custom_cycler})

    fig = main(filename, groups)
    filename = figures_dir / f"panel_boxplot{suffix}.pdf"
    fig.savefig(filename)
    print("Saved to: ", filename)

    print("\nDone!\n")
