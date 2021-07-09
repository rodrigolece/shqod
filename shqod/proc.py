"""Calculate features from paths."""

from typing import Tuple, List
from .io import UntidyLoader, read_level_grid
from .matrices import od_matrix, mobility_functional, visiting_order
from .utils import path_length, path_curvature, path_bdy
from .smoothing import smooth

import os
import itertools
from functools import lru_cache
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from scipy.spatial import distance_matrix
import pandas as pd
from esig import tosig as pathsig


class TrajProcessor(object):
    def __init__(self, level: int, gender: str, **kwargs):

        self.level = level
        self.country = "uk"  # currently not used
        self.gender = gender

        #  assert "max_sigdim" in kwargs, "hyperparameter needed: 'max_sigdim'"
        #  assert "spline_res" in kwargs, "hyperparameter needed: 'spline_res'"
        #  assert "grid_dir" in kwargs, "hyperparameter needed: 'grid_dir'"

        #  self.max_sigdim = kwargs["max_sigdim"]
        #  self.spline_res = kwargs["spline_res"]

        for key, val in kwargs.items():
            setattr(self, key, val)

        if "grid_dir" in kwargs:
            grid_dir = kwargs["grid_dir"]
            file = os.path.join(grid_dir, f"level{level:02}.json")
            coords, wd, lg = read_level_grid(file)
            self.grid_coords = coords
            self.grid_width = wd
            self.grid_length = lg

    def __str__(self):
        return f"Processor: {self.country} - {self.gender} - {self.level}"

    def get_len(self, path):
        return path_length(path)

    def get_curv(self, path):
        return path_curvature(path)

    def get_bdy(self, path):
        return path_bdy(path, self.grid_coords)

    def get_sig(self, path):
        max_sigdim = getattr(self, "max_sigdim")

        if max_sigdim is None:
            raise ValueError("hyperparameter needed: 'max_sigdim'")

        return pathsig.stream2logsig(path, self.max_sigdim)

    def get_smooth_features(self, df, feat_types, keys=["id"]):
        # TODO: some sort of check for the feat_types
        out = df.reset_index(drop=True)
        N = len(out)

        sig_flag = False

        if "sig" in feat_types:
            sig_flag = True
            sig_arr = np.zeros((N, 8))  # TODO: always 8 or depends of hp?
            feat_types = feat_types.copy()
            feat_types.pop(feat_types.index("sig"))

        methods = [getattr(self, f"get_{feat}") for feat in feat_types]
        methods = list(filter(None.__ne__, methods))

        arr = np.zeros((N, len(methods)))
        cols = feat_types.copy()

        for i, row in out.iterrows():
            path = smooth(row.trajectory_data)
            arr[i] = [method(path) for method in methods]

            if sig_flag:
                sig_arr[i] = self.get_sig(path)

        if sig_flag:
            arr = np.hstack((arr, sig_arr))
            cols += ["sig" + str(i) for i in range(1, 9)]  # TODO: same, 8?

        results_df = pd.DataFrame(arr, columns=cols).join(out[keys])
        out = out.drop(columns="trajectory_data")

        return out.merge(results_df, on=keys)

    def get_vo(self, path):
        flag_coords = getattr(self, "flag_coords")
        R = getattr(self, "R")

        if flag_coords is None or R is None:
            raise ValueError("hyperparameters needed: 'flag_coords' and 'R'")

        # safe_mode=False for no warnings
        return visiting_order(path, flag_coords, R=R, safe_mode=False)


class NormativeProcessor(TrajProcessor):
    def __init__(self, loader: UntidyLoader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loader = loader

        assert "window_size" in kwargs, "hyperparameter needed: 'window_size'"
        assert "weight_scale" in kwargs, "hyperparameter needed: 'weight_scale'"

        self.window_size = kwargs["window_size"]
        self.weight_scale = kwargs["weight_scale"]

        key = (self.level, self.gender)
        if key not in loader.loaded:
            loader.get(*key)  # pre-load the df

        # This is used to store a particular instance of normative od matrix
        # as a tuple (mat, N) where N is the number of paths used
        self._normative_mat = None

        return None

    def __str__(self):
        return f"Normative processor: {self.country} - {self.gender} - {self.level}"

    @property
    def normative_mat(self):
        return self._normative_mat

    @normative_mat.setter
    def normative_mat(self, mat):
        assert sp.isspmatrix_csr(mat), "error: invalid format for matrix"
        wd, lg = self.grid_width, self.grid_length
        N = wd * lg
        assert mat.shape == (N, N), "error: invalid dimensions for matrix"
        self._normative_mat = mat
        return None

    def _get_df_for_age(self, age: int):
        df = self.loader.get(self.level, self.gender, age)
        self.loader.json_to_array()  # we make sure to always work with arrays
        return df

    @lru_cache
    def normative_od_matrix_for_age(self, age: int):
        wd, lg = self.grid_width, self.grid_length

        df = self._get_df_for_age(age)
        N = len(df)

        lex_ts = [wd * t[:, 1] + t[:, 0] for t in df.trajectory_data]
        od_mat = od_matrix(lex_ts, wd * lg)

        return od_mat, N

    @lru_cache
    def normative_od_matrix_window(self, centre):
        window = self.window_size
        scale = self.weight_scale

        ages = range(centre - window, centre + window + 1)

        if scale == np.inf:
            weights = np.ones(len(ages))
        else:
            weights = st.distributions.norm(centre, scale=scale).pdf(ages)
            weights /= weights.sum()

        wd, lg = self.grid_width, self.grid_length
        N = wd * lg
        out = sp.csr_matrix((N, N))

        for i, age in enumerate(ages):
            mat, N = self.normative_od_matrix_for_age(age)
            out += weights[i] * mat / N

        return out

    def _od_matrix_from_path(self, path):
        if self.normative_mat is None:
            raise Exception("normative OD matrix has not been set")

        wd, lg = self.grid_width, self.grid_length
        lex = path[:, 1] * wd + path[:, 0]

        return od_matrix([lex], wd * lg)

    def get_fro(self, path):
        if self.normative_mat is None:
            raise Exception("normative OD matrix has not been set")

        od_mat = self._od_matrix_from_path(path)
        return np.linalg.norm((self.normative_mat - od_mat).toarray(), "fro")

    def get_sup(self, path):
        if self.normative_mat is None:
            raise Exception("normative OD matrix has not been set")

        od_mat = self._od_matrix_from_path(path)
        return np.linalg.norm((self.normative_mat - od_mat).toarray(), np.inf)

    def get_match(self, path):
        if self.normative_mat is None:
            raise Exception("normative OD matrix has not been set")

        od_mat = self._od_matrix_from_path(path)
        r, s = od_mat.nonzero()
        return self.normative_mat[r, s].sum() / len(r)

    def get_mob(self, path):
        if self.normative_mat is None:
            raise Exception("normative OD matrix has not been set")

        wd = self.grid_width
        return mobility_functional(path, self.normative_mat, wd)

    def get_coarse_features(self, df, feat_types, keys=["id"]):
        if self.normative_mat is None:
            raise Exception("normative OD matrix has not been set")

        # TODO: some sort of check for the feat_types
        out = df.reset_index(drop=True)
        N = len(out)

        methods = [getattr(self, f"get_{feat}") for feat in feat_types]
        methods = list(filter(None.__ne__, methods))

        arr = np.zeros((N, len(methods)))
        cols = feat_types.copy()

        for i, row in out.iterrows():
            path = row.trajectory_data
            arr[i] = [method(path) for method in methods]

        results_df = pd.DataFrame(arr, columns=cols).join(out[keys])
        out = out.drop(columns="trajectory_data")

        return out.merge(results_df, on=keys)

    def get_windowed_features(self, df, feat_types, keys=["id"], sort_id=False):
        # TODO: some sort of check for feat_types

        gby = df.groupby("age")
        out = []

        for age, age_df in gby:
            wmat = self.normative_od_matrix_window(age)
            self._normative_mat = wmat  # we skip tests
            out.append(self.get_coarse_features(age_df, feat_types, keys=keys))

        if sort_id:
            out = pd.concat(out).sort_values("id").reset_index(drop=True)
        else:
            out = pd.concat(out, ignore_index=True)

        return out


def compute_percentiles(
    df: pd.DataFrame, loader: UntidyLoader, feat_types: List[str], drop_sig: bool = True
) -> pd.DataFrame:
    """
    Compute the percentile score for a set of features and a reference pop.

    Parameters
    ----------
    df : pd.DataFrame
        The input data; it should contain the calculated features.
    loader : UntidyLoader
       The loader that get the DataFrames for the normative population; these
       should also contain calculated features.
    feat_types : List[str]
        The names of the features columns to use.
    drop_sig: bool, optional
        Whether to ignore path signature features (default is True). This
        option overrides any names that could be contained inside the list
        `feat_types`.

    Returns
    -------
    pd.DataFrame
        The output mimics the shape of the input `df` with the original entries
        replaced by numbers in [0, 100] that represent the percentiles.

    """
    if drop_sig:  # do not compute percentiles for path signature
        drop_cols = filter(lambda c: c.startswith("sig"), df.columns)
        out = df.drop(columns=drop_cols)
    else:
        out = df

    # the features for which a high value is bad need to be reversed
    reverse_cols = ["len", "curv", "bdy", "fro", "sup"]

    levels = df.level.unique()
    genders = df.gender.unique()

    # percentile computation for each level and gender
    for lvl, g in itertools.product(levels, genders):
        idx = (out.level == lvl) & (out.gender == g)

        for i, row in out.loc[idx].iterrows():
            ref = loader.get(lvl, g, row.age)  # for each age

            for k, col in enumerate(feat_types):
                scores, val = ref[col], row[col]
                if col in reverse_cols:  # reverse the scores
                    scores = -scores
                    val = -val

                out.loc[i, col] = st.percentileofscore(scores, val, kind="weak")
                # weak corresponds to the CDF definition

    return out
