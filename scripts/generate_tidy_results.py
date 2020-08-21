import os
import argparse
import string
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from shqod import (
    trajecs_from_files,
    read_level_grid,
    od_matrix,
    calculate_field,
    field_to_dict,
    mobility_functional,
    fractalD,
    trajectory_length
)


project_dir = os.environ['dementia']  # set in the shell

grids_dir = os.path.join(project_dir, 'data', 'raw', 'grids', '')

processed_dir = os.path.join(project_dir, 'data', 'processed', '')
apoe_dir = os.path.join(project_dir, 'data', 'apoe')
e3e4_dir = os.path.join(apoe_dir, 'e3e4')
e3e3_dir = os.path.join(apoe_dir, 'e3e3')
e4e4_dir = os.path.join(apoe_dir, 'e4e4')


def run_tests(filenames, mat, N, width, length):
    """Run the battery of 6 tests."""

    test_names = ['Frob. norm', 'Inf. norm', 'Restrict. sum',
                  'Mobty functional', 'Fractal dim.', 'Tot. length']

    n = len(filenames)
    out = np.ones((n, 6))

    ts = [list(t) for t in trajecs_from_files(filenames, lexico=False)]
    lex_ts = trajecs_from_files(filenames, lexico=True, grid_width=width)
    group_mats = [od_matrix([t], width * length) for t in lex_ts]

    norm_mat = mat/N
    # F_dict = field_to_dict(*calculate_field(mat, width, nb_trajecs=N))

    for i, m in enumerate(group_mats):
        # Matrix differences
        fro = np.linalg.norm((norm_mat - m).toarray(), 'fro')
        inf = np.linalg.norm((norm_mat - m).toarray(), np.inf)

        # Sum of matching entries
        r, s = m.nonzero()
        match = mat[r, s].sum() / (N*len(r))

        # Mobility functional
        mob = mobility_functional(ts[i], mat, width, N)

        # Fractal dimension
        d = fractalD(ts[i], width, length)

        # Total length
        l = trajectory_length(ts[i])

        out[i] = [fro, inf, match, mob, d, l]

    return out, test_names


def level_tidy_df(group_dirs, level):
    """Load level data for several groups and run the tests."""

    # Load the level dimensions 
    _, _, width, length = read_level_grid(grids_dir + f'level{level:02}.json')

    # Load the OD matrices combined for m/f and ages 51-70
    Rs = ['5160', '6170']  # TODO: take the correct age ranges
    N = 0

    L = width * length
    mat = sp.csr_matrix((L, L))

    for age_range in Rs:
        # the nbr of entries in that age range
        counts_df = pd.read_csv(processed_dir + 'uk_counts.csv')
        idx = (counts_df.level == level) & (counts_df.age_range == age_range)
        N += counts_df.loc[idx]['counts'].sum()

        f = os.path.join(processed_dir, f'level_{level}_uk_{age_range}_f.npz')
        m = os.path.join(processed_dir, f'level_{level}_uk_{age_range}_m.npz')
        fmat = sp.load_npz(f)
        mmat = sp.load_npz(m)

        mat += fmat + mmat

    # Load the trajectories for each group and calculate the results
    dfs = []

    for i, d in enumerate(group_dirs):
        filenames = [os.path.join(root, name)
                     for root, dirs, files in os.walk(d)
                     for name in files
                     if name.startswith(f'level0{level:02}')]

        res, test_names = run_tests(filenames, mat, N, width, length)
        df = pd.DataFrame(res, columns=test_names)
        df['group'] = string.ascii_uppercase[i]  # no more than 26 groups!
        dfs.append(df)

    return pd.concat(dfs)


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
    assert filename.endswith('.csv'), 'input should be csv file'

    # below would be calculated using the name of the groups
    # (i.e. we would use the names to cosntruct the dirs)
    populations = (e3e4_dir, e3e3_dir, e4e4_dir)
    dfs = []

    print('Calculating level results')
    for l in tqdm(levels):
        df = level_tidy_df(populations, l)
        df['level'] = l
        dfs.append(df)

    tidy_df = pd.concat(dfs)
    tidy_df.to_csv(filename, index=False)

    print('\nDone!\n')
