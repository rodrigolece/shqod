"""shqod - Origin-Destination tools to work with the SHQ dataset."""

from shqod.io import (
    LevelsLoader,
    read_path_csv,
    read_path_feather,
    read_level_grid,
    read_level_size,
    read_level_flags,
    write_feather,
)

from shqod.proc import (
    AbsProcessor,
    RelProcessor,
    compute_percentiles,
    fill_missing_attempts,
    compute_pvalues,
    compute_auc,
    norm,
)

from shqod.matrices import (
    od_matrix,
    mobility_field,
    #  od_matrix_brokenup,
    #  breakup_by_flags,
)

from shqod.paths import (
    smooth,
    visiting_order,
    vo_correctness,
    path_length,
    avg_curvature,
    bdy_affinity,
    #  fractal_dim,
    frobenius_deviation,
    supremum_deviation,
    sum_match,
    mobility_functional,
)
