"""Calculate features from trajectories."""

from typing import Tuple
from .io import UntidyLoader, read_level_grid, trajecs_from_df
from .matrices import od_matrix, mobility_functional
from .utils import path_length, path_curvature, path_dtb

import os
import numpy as np
import scipy.sparse as sp
from esig import tosig as pathsig


class TrajProcessor(object):
    def __init__(self,
                 level: int,
                 gender: str,
                 **kwargs):

        self.level = level
        self.country = 'uk'   # currently not used
        self.gender = gender

        assert 'max_sigdim' in kwargs, "hyperparameter needed: 'max_sigdim'"
        assert 'spline_res' in kwargs, "hyperparameter needed: 'spline_res'"
        assert 'grid_dir' in kwargs, "hyperparameter needed: 'grid_dir'"

        self.max_sigdim = kwargs['max_sigdim']
        self.spline_res = kwargs['spline_res']

        # Currently not needed, but perhaps later
        # for key, val in kwargs.items():
        #     setattr(self, key, val)

        if 'grid_dir' in kwargs:
            grid_dir = kwargs['grid_dir']
            file = os.path.join(grid_dir, f'level{level:02}.json')
            coords, wd, lg = read_level_grid(file)
            self.grid_coords = coords
            self.grid_width = wd
            self.grid_length = lg

    def __str__(self):
        return f'Processor: {self.country} - {self.gender} - {self.level}'

    def get_len(self, trajec):
        return path_length(trajec)

    def get_curv(self, trajec):
        return path_curvature(trajec)

    def get_sig(self, trajec):
        return pathsig.stream2logsig(trajec, self.max_sigdim)

    def get_dtb(self, trajec):
        return path_dtb(trajec, self.grid_coords)


class NormativeProcessor(TrajProcessor):
    def __init__(self, loader: UntidyLoader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loader = loader

        key = (self.level, self.gender)
        if key not in loader.loaded:
            loader.get(*key, 19)  # arbitrary age simply to load the df

        # This is used to store a particular instance of normative od matrix
        # as a tuple (mat, N) where N is the number of trajectories used
        self._normative_mat = None

        return None

    @property
    def normative_mat(self):
        return self._normative_mat

    @normative_mat.setter
    def normative_mat(self, mat_N: Tuple[sp.csr.csr_matrix, int]):
        mat, N = mat_N
        assert N > 0, 'expected positve N'
        self._normative_mat = mat_N
        self.normalised_mat = mat / N

        return None

    def _get_df_for_age(self, age: int):
        return self.loader.get(self.level, self.gender, age)

    def normative_od_matrix_for_age(self, age: int):
        wd, lg = self.grid_width, self.grid_length

        df = self._get_df_for_age(age)
        N = len(df)

        lex_ts = trajecs_from_df(df, lexico=True, grid_width=wd)
        od_mat = od_matrix(lex_ts, wd*lg)

        return od_mat, N

    def _od_matrix_from_trajec(self, trajec):
        if self._normative_mat is None:
            raise Exception('normative OD matrix has not been set')

        wd, lg = self.grid_width, self.grid_length
        lex = trajec[:, 1] * wd + trajec[:, 0]

        return od_matrix([lex], wd*lg)

    def get_fro(self, trajec):
        if self._normative_mat is None:
            raise Exception('normative OD matrix has not been set')

        od_mat = self._od_matrix_from_trajec(trajec)
        return np.linalg.norm((self.normalised_mat - od_mat).toarray(), 'fro')

    def get_inf(self, trajec):
        if self._normative_mat is None:
            raise Exception('normative OD matrix has not been set')

        od_mat = self._od_matrix_from_trajec(trajec)
        return np.linalg.norm((self.normalised_mat - od_mat).toarray(), np.inf)

    def get_sum_match(self, trajec):
        if self._normative_mat is None:
            raise Exception('normative OD matrix has not been set')

        od_mat = self._od_matrix_from_trajec(trajec)
        r, s = od_mat.nonzero()
        return self.normalised_mat[r, s].sum() / len(r)

    def get_mob(self, trajec):
        if self._normative_mat is None:
            raise Exception('normative OD matrix has not been set')

        wd = self.grid_width
        normative_mat, N = self.normative_mat

        return mobility_functional(trajec, normative_mat, wd, N)
