"""Calculate features from paths."""

import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.stats as st
import pandas as pd

# from esig import tosig

from shqod.io import LevelsLoader, read_level_grid, read_level_flags
from shqod.matrices import od_matrix, mobility_field
from shqod.paths import (
    path_length,
    avg_curvature,
    bdy_affinity,
    frobenius_deviation,
    supremum_deviation,
    sum_match,
    mobility_functional,
    visiting_order,
    smooth,
)


class AbsProcessor(object):
    def __init__(self, level: int, gender: str, **kwargs):

        self.level = level
        self.country = "uk"  # currently not used
        self.gender = gender

        #  assert "spline_res" in kwargs, "hyperparameter needed: 'spline_res'"
        #  self.spline_res = kwargs["spline_res"]

        for key, val in kwargs.items():
            setattr(self, key, val)

        if "grid_dir" in kwargs:
            grid_dir = Path(kwargs["grid_dir"])

            file = grid_dir / f"level{level:02}.json"
            coords, wd, lg = read_level_grid(file)
            self.grid_coords = coords
            self.grid_size = (wd, lg)
            self.grid_width = wd
            self.grid_length = lg

            self.flag_coords = read_level_flags(file)[::-1]
            # NB: for levels 6, 8 and 11 I've tested the flags are in the reversed
            # order, but I don't know if this will always hold

            inner_bdy_file = grid_dir / f"inner_bdy_level{level:02}.npy"
            if os.path.isfile(inner_bdy_file):
                self.inner_bdy = smooth(np.load(inner_bdy_file))

    def __str__(self):
        return f"Processor: {self.country} - {self.gender} - {self.level}"

    def get_len(self, path):
        return path_length(path)

    def get_curv(self, path):
        return avg_curvature(path)

    def get_bdy(self, path):
        scale = getattr(self, "bdy_scale", 4.0)
        rin = getattr(self, "bdy_rin", None)
        rout = getattr(self, "bdy_rout", None)

        if rin is None:
            raise ValueError("hyperparameter needed: 'rin'")
        elif rout is None:
            raise ValueError("hyperparameter needed: 'rout'")

        return bdy_affinity(path, self.inner_bdy, rin=rin, rout=rout, scale=scale)

    # def get_sig(self, path):
    #     max_sigdim = getattr(self, "max_sigdim", None)

    #     if max_sigdim is None:
    #         raise ValueError("hyperparameter needed: 'max_sigdim'")

    #     return tosig.stream2logsig(path, self.max_sigdim)

    def get_smooth_features(self, df, feat_types, keys=["id"]):
        #  feat_types = _get_iterable(feat_types)
        # TODO: not working because of copy below
        # TODO: some sort of check for the feat_type's
        out = df.reset_index(drop=True)
        N = len(out)

        sig_flag = False

        if "sig" in feat_types:
            # TODO: remove expansion altogether, store as array
            sig_flag = True
            sig_arr = np.zeros((N, 8))
            feat_types = feat_types.copy()
            feat_types.pop(feat_types.index("sig"))

        methods = [getattr(self, f"get_{feat}") for feat in feat_types]
        methods = list(filter(None.__ne__, methods))  # filter out None

        arr = np.zeros((N, len(methods)))
        cols = feat_types.copy()

        for i, row in out.iterrows():
            path = smooth(row.trajectory_data)
            arr[i] = [method(path) for method in methods]

            if sig_flag:
                sig_arr[i] = self.get_sig(path)

        if sig_flag:
            # TODO: remove expansion altogether, store as array
            arr = np.hstack((arr, sig_arr))
            cols += ["sig" + str(i) for i in range(1, 9)]

        results_df = pd.DataFrame(arr, columns=cols).join(out[keys])
        out = out.drop(columns="trajectory_data")

        return out.merge(results_df, on=keys)

    def get_vo(self, path):
        flag_coords = getattr(self, "flag_coords", None)

        if flag_coords is None:
            raise ValueError("hyperparameter needed: 'flag_coords'")

        params = {"safe_mode": False}  # for no warnings
        R = getattr(self, "R", None)
        if R:
            params.update({"R": R})

        return visiting_order(path, flag_coords, **params)


class RelProcessor(AbsProcessor):
    def __init__(self, loader: LevelsLoader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loader = loader

        assert "window_size" in kwargs, "hyperparameter needed: 'window_size'"
        assert "weight_scale" in kwargs, "hyperparameter needed: 'weight_scale'"

        self.window_size = kwargs["window_size"]
        self.weight_scale = kwargs["weight_scale"]

        key = (self.level, self.gender)
        if key not in loader.loaded:
            loader.get(*key)  # pre-load the df

        self._normative_mat = None
        self._normative_field = None

    def __str__(self):
        return f"Normative processor: {self.country} - {self.gender} - {self.level}"

    @property
    def normative_mat(self):
        return self._normative_mat

    @property
    def normative_field(self):
        return self._normative_field

    @normative_mat.setter
    def normative_mat(self, mat):
        assert sp.isspmatrix_csr(mat), "error: invalid format for matrix"
        size = np.multiply(*self.grid_size)
        assert mat.shape == (size, size), "error: invalid dimensions for matrix"
        self._normative_mat = mat
        self._normative_field = mobility_field(mat, self.grid_width)

    def _get_df_for_age(self, age: int):
        df = self.loader.get(self.level, self.gender, age)
        return df

    @lru_cache
    def _normative_od_matrix_for_age(self, age: int):
        df = self._get_df_for_age(age)
        N = len(df)

        return od_matrix(df.trajectory_data, self.grid_size), N

    @lru_cache
    def normative_od_matrix_windowed(self, centre):
        window = self.window_size
        scale = self.weight_scale

        ages = range(centre - window, centre + window + 1)

        if scale == np.inf:
            weights = np.ones(len(ages))
        else:
            weights = st.distributions.norm(centre, scale=scale).pdf(ages)
            weights /= weights.sum()

        size = np.multiply(*self.grid_size)
        out = sp.csr_matrix((size, size))

        for i, age in enumerate(ages):
            mat, N = self._normative_od_matrix_for_age(age)
            out += weights[i] * mat / N

        return out

    def get_fro(self, path):
        return frobenius_deviation(path, self.grid_size, self.normative_mat)

    def get_sup(self, path):
        return supremum_deviation(path, self.grid_size, self.normative_mat)

    def get_match(self, path):
        return sum_match(path, self.grid_size, self.normative_mat)

    def get_mob(self, path):
        return mobility_functional(path, self.normative_field)

    def get_coarse_features(self, df, feat_types, keys=["id"]):
        #  feat_types = _get_iterable(feat_types)
        # TODO: not working because of copy below
        # TODO: some sort of check for the feat_types
        if self.normative_mat is None:
            raise Exception("normative OD matrix has not been set")

        out = df.reset_index(drop=True)
        N = len(out)

        vo_flag = False

        if "vo" in feat_types:
            vo_flag = True
            vo_list = []
            feat_types = feat_types.copy()
            feat_types.pop(feat_types.index("vo"))

        methods = [getattr(self, f"get_{feat}") for feat in feat_types]
        methods = list(filter(None.__ne__, methods))

        arr = np.zeros((N, len(methods)))
        cols = feat_types.copy()

        for i, row in out.iterrows():
            path = row.trajectory_data
            arr[i] = [method(path) for method in methods]

            if vo_flag:
                vo_list.append(self.get_vo(path))

        results_df = pd.DataFrame(arr, columns=cols).join(out[keys])

        if vo_flag:
            results_df["vo"] = vo_list

        out = out.drop(columns="trajectory_data")

        return out.merge(results_df, on=keys)

    def get_windowed_features(self, df, feat_types, keys=["id"], sort_id=False):
        # TODO: some sort of check for feat_types

        gby = df.groupby("age")
        out = []

        for age, age_df in gby:
            wmat = self.normative_od_matrix_windowed(age)
            self.normative_mat = wmat  # this also sets normative_field
            out.append(self.get_coarse_features(age_df, feat_types, keys=keys))

        if sort_id:
            out = pd.concat(out).sort_values("id").reset_index(drop=True)
        else:
            out = pd.concat(out, ignore_index=True)

        return out
