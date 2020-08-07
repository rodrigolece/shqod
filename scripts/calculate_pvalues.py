import argparse
import pandas as pd
import scipy.stats as st
from tabulate import tabulate


def pvalues(df, correct_gameplay=False):

    levels = set(df.level)
    # n = len(filenames_one)
    # assert len(filenames_two) == n  # by desing we have the same participants

    if correct_gameplay:
        # We use the results for the first two levels to correct for gameplay
        assert 1 in levels and 2 in levels

        one = df.loc[df.level == 1].drop(columns=['group', 'level'])
        two = df.loc[df.level == 2].drop(columns=['group', 'level'])
        sum_df = one + two.values  # .values for elementwise ignoring indices
        cols = df.drop(columns=['group', 'level']).columns

        correc_df = df.copy()

        for l in levels.difference((1, 2)):
            idx = correc_df.level == l
            correc_df.loc[idx, cols] = df.loc[idx, cols] / sum_df.values
    else:
        correc_df = df

    # p-values
    level_pvals = []
    for l in levels.difference((1, 2)):
        a = correc_df.loc[(correc_df.level == l) & (correc_df.group == 'A')]
        b = correc_df.loc[(correc_df.level == l) & (correc_df.group == 'B')]

        pvals = [st.ttest_ind(a.values[:, i],
                              b.values[:, i],
                              equal_var=False).pvalue
                 for i in range(6)]
        level_pvals.append([l] + pvals)

    return level_pvals


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='file containing tidy data')
    parser.add_argument('-g', '--groups', default='AB',
                        help='the label of two groups to compare')
    args = parser.parse_args()

    filename = args.input_file
    groups = args.groups.upper()
    assert filename.endswith('.csv'), 'input should be csv file'
    assert len(groups) == 2, 'should give two groups'

    # Read the input data
    results_df = pd.read_csv(filename)
    idx = results_df.group.isin([*groups])
    results_df = results_df.loc[idx]

    # Calculate the p-values with and without correction
    pvals = pvalues(results_df, correct_gameplay=False)
    correc_pvals = pvalues(results_df, correct_gameplay=True)

    head = ['Level', 'Frob. norm', 'Inf. norm', 'Restrict. sum',
            'Mobty functional', 'Fractal dim.', 'Tot. length']

    print('\np-values, no gameplay correction')
    print(tabulate(pvals, headers=head, floatfmt='.3f'))

    print('\np-values, with gameplay correction')
    print(tabulate(correc_pvals, headers=head, floatfmt='.3f'))

    print('\nDone!\n')
