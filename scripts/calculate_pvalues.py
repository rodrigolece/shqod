import argparse
import pandas as pd
import scipy.stats as st
from tabulate import tabulate


def pvalues(df, groups, correct_gameplay=False):
    """Calculate p-values for the given groups."""

    g1, g2 = groups
    levels = set(df.level)

    # The test names come directly from generate_tidy_results.py
    test_names = [
        'Frob. norm', 'Inf. norm', 'Restrict. sum',
        'Mobty functional', 'Fractal dim.', 'Tot. length'
    ]

    if correct_gameplay:
        # We use the results for the first two levels to correct for gameplay
        assert 1 in levels and 2 in levels

        one = df.loc[df.level == 1][test_names]
        two = df.loc[df.level == 2][test_names]
        sum_df = one + two.values  # .values for elementwise ignoring indices

        correc_df = df.copy()

        for l in levels.difference((1, 2)):
            idx = correc_df.level == l
            correc_df.loc[idx, test_names] = df.loc[idx, test_names] / sum_df.values
    else:
        correc_df = df

    # p-values
    level_pvals = []
    for lvl in levels.difference((1, 2)):
        lvl_idx = correc_df.level == lvl
        first = correc_df.loc[lvl_idx & (correc_df.group == g1)]
        second = correc_df.loc[lvl_idx & (correc_df.group == g2)]

        pvals = [st.ttest_ind(first[test_names].values[:, i],
                              second[test_names].values[:, i],
                              equal_var=False).pvalue
                 for i in range(6)]
        level_pvals.append([lvl] + pvals)

    return level_pvals


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='file containing tidy data')
    parser.add_argument('-g', '--groups', nargs='+',
                        default=['e3e4', 'e3e3'],
                        help='the label of two groups to compare')
    args = parser.parse_args()

    filename = args.input_file
    groups = args.groups
    assert filename.endswith('.csv'), 'input should be csv file'
    assert len(groups) == 2, 'should give two groups'

    # Read the input data
    results_df = pd.read_csv(filename)
    idx = results_df.group.isin([*groups])
    results_df = results_df.loc[idx]

    # Calculate the p-values with and without correction
    pvals = pvalues(results_df, groups, correct_gameplay=False)
    correc_pvals = pvalues(results_df, groups, correct_gameplay=True)

    head = ['Level', 'Frob. norm', 'Inf. norm', 'Restrict. sum',
            'Mobty functional', 'Fractal dim.', 'Tot. length']

    print('\np-values, no gameplay correction')
    print(tabulate(pvals, headers=head, floatfmt='.3f'))

    print('\np-values, with gameplay correction')
    print(tabulate(correc_pvals, headers=head, floatfmt='.3f'))

    print('\nDone!\n')
