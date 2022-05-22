import os
from pathlib import Path
import itertools
import functools
import multiprocessing
import pickle

from tqdm import tqdm

from shqod import utils, LevelsLoader, RelProcessor


data_dir = Path(os.environ["dementia"]) / "data"
normative_dir = data_dir / "normative"

hp = {
    "grid_dir": data_dir / "maps",
    "window_size": 5,
    "weight_scale": 2.0,  # np.inf
}


def process_level_gender(loader, key, age="50:80"):  # 80
    low, high = utils.parse_age(age)
    proc = RelProcessor(loader, *key, **hp)

    out = {}

    for age in range(low, high + 1):
        out[(*key, age)] = proc.normative_od_matrix_windowed(age)

    return out


def precompute_all(loader, levels, genders, age="50:80"):

    part = functools.partial(process_level_gender, loader, age=age)
    iterable = itertools.product(levels, genders)

    with multiprocessing.Pool() as p:
        results = list(tqdm(p.imap(part, iterable), total=nb_iters))

    # My implementation of reduce
    out = {}

    for d in results:
        out.update(d)

    return out


if __name__ == "__main__":

    save = False

    levels = [6, 8, 11]  # 1, 2
    genders = ["f", "m"]
    nb_iters = len(levels) * len(genders)

    loader = LevelsLoader(normative_dir / "paths", fmt="feather")
    results = precompute_all(loader, levels, genders, age="50:80")

    filename = normative_dir / "od_mats.pkl"
    if save:
        with open(filename, "wb") as f:
            pickle.dump(results, f)

        print("Saved results to: ", filename)

    print("\nDone!\n")
