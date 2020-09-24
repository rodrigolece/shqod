import os
import argparse
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from shqod import (
    read_trajec_csv,
    trajec,
    read_level_grid,
    od_matrix,
    calculate_field,
    field_to_dict,
    mobility_functional,
    fractalD,
    trajectory_length
)


project_dir = os.environ['dementia']  # set in the shell

grids_dir = os.path.join(project_dir, 'data', 'grids', '')

processed_dir = os.path.join(project_dir, 'data', 'processed', '')
apoe_dir = os.path.join(project_dir, 'data', 'apoe_dataframes')
ad_dir = os.path.join(project_dir, 'data', 'ad_dataframe')
# e3e4_dir = os.path.join(apoe_dir, 'e3e4')
# e3e3_dir = os.path.join(apoe_dir, 'e3e3')
# e4e4_dir = os.path.join(apoe_dir, 'e4e4')


class NormativeBenchmark(object):

    def __init__(self,
                 age_range: str,
                 gender: str,
                 od_matrix: sp.csr.csr_matrix,
                 nb_trajecs: int,
                 grid_width: int,
                 grid_length: int,
                 level: int = None,
                 flags: np.array = None):

        self.age_range = age_range
        self.gender = gender
        self.normative_mat = od_matrix
        self.N = nb_trajecs
        self.width = grid_width
        self.length = grid_length

        if level:
            self.level = level

        if flags:
            self.flags = flags

        self.test_names = [
            'Frob. norm', 'Inf. norm', 'Restrict. sum',
            'Mobty functional', 'Fractal dim.', 'Tot. length'
        ]

    def __str__(self):
        return f'Normative bechmark - level {self.level} - {self.age_range} - {self.gender} (N={self.N})'

    def test_trajectory(self, trajectory_data: str):
        """Run the battery of 6 tests."""

        t = list(trajec(trajectory_data, lexico=False))
        lex = trajec(trajectory_data, lexico=True, grid_width=self.width)
        od_mat = od_matrix([lex], self.width * self.length)

        norm_mat = self.normative_mat / self.N

        fro = np.linalg.norm((norm_mat - od_mat).toarray(), 'fro')
        inf = np.linalg.norm((norm_mat - od_mat).toarray(), np.inf)

        # Sum of matching entries
        r, s = od_mat.nonzero()
        match = norm_mat[r, s].sum() / len(r)

        # Mobility functional
        mob = mobility_functional(t, self.normative_mat, self.width, self.N)

        # Fractal dimension
        dim = fractalD(t, self.width, self.length)

        # Total length
        lgt = trajectory_length(t)

        return [fro, inf, match, mob, dim, lgt]


def load_environments(levels, Rs):
    """Load the normative benchmark environments."""

    counts_df = pd.read_csv(processed_dir + 'uk_counts.csv')

    out = dict()

    for lvl in levels:
        # Level width and length
        filename_grid = grids_dir + f'level{lvl:02}.json'
        _, _, width, length = read_level_grid(filename_grid)

        # The table containing counts per level, age group and gender
        lvl_counts_df = counts_df.loc[counts_df.level == lvl]

        for age_range, gender in itertools.product(Rs, ['f', 'm']):
            # The processed normative matrix
            filename = os.path.join(
                processed_dir,
                f'level_{lvl}_uk_{age_range}_{gender}.npz'
            )
            norm_mat = sp.load_npz(filename)

            # The nbr of entries for that age range and gender
            age_counts = (lvl_counts_df
                          .loc[counts_df.age_range == age_range]
                          .set_index('gender')['counts']
                          .to_dict())
            N = age_counts[gender]

            out[(lvl, age_range, gender)] = NormativeBenchmark(
                age_range, gender, norm_mat, N, width, length, level=lvl
            )

    return out


def group_tidy_df(filename, envs, Rs):
    """Load level data for several groups and run the tests."""

    df = read_trajec_csv(filename)
    levels = df.level.unique()

    # We append the age range
    df['age_range'] = (
        df['age']
        .apply(lambda x: x // 10)
        .replace(dict(zip([int(x[0]) for x in Rs], Rs)))
    )

    # The test names are defined inside the environment object
    test_names = envs[list(envs)[0]].test_names

    dfs = []

    for lvl in tqdm(levels):

        lvl_df = df.loc[df.level == lvl].reset_index(drop=True)
        results = np.ones((len(lvl_df), 6))

        # Load the trajectories and calculate the results
        for k, row in lvl_df.iterrows():
            ar, g, data = row[['age_range', 'gender', 'trajectory_data']]
            results[k] = envs[(lvl, ar, g)].test_trajectory(data)

        res_df = pd.DataFrame(results, columns=test_names)
        res_df['level'] = lvl
        dfs.append(res_df)

    # We keep original ids in the results df
    out_df = pd.concat(dfs).join(df['id'])

    return out_df[['id', 'level'] + test_names]


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', help='name of output file')
    parser.add_argument('-l', '--levels', default=[1, 2, 6, 8, 11],
                        help='the levels to be used')
    # parser.add_argument('-g', '--groups', default='AB',
    #                     help='the label of two groups to compare')
    args = parser.parse_args()

    filename = args.output_file
    levels = args.levels
    # groups = args.groups
    assert filename.endswith('.csv'), 'output should be csv file'

    # The three genetic groups of the apoe study
    populations = [
        os.path.join(apoe_dir, f)
        for f in os.listdir(apoe_dir)
    ]
    populations += [os.path.join(ad_dir, 'ad.csv')]

    # We know beforehand the age ranges for the participants of the study
    Rs = ['5160', '6170', '71plus']

    envs = load_environments(levels, Rs)

    dfs = []

    for group in populations:
        name = os.path.basename(group).replace('apoe_', '').replace('.csv', '')
        print(f'Current group is: {name}')

        df = group_tidy_df(group, envs, Rs)
        df['group'] = name
        dfs.append(df)
        print()

    tidy_df = pd.concat(dfs)
    tidy_df.to_csv(filename, index=False)

    print('Done!\n')
