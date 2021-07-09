"""shqod - Origin-Destination tools to work with the SHQ dataset."""

from .io import (
    TidyLoader,
    UntidyLoader,
    read_path_csv,
    read_path_feather,
    previous_attempts,
    duplicated_attempts,
    path,
    paths_from_df,
    paths_from_files,
    read_level_grid,
)

from .matrices import (
    od_matrix,
    od_matrix_brokenup,
    breakup_array_by_flags,
    reduce_matrix,
    calculate_field,
    field_to_dict,
    mobility_functional,
)

from .proc import TrajProcessor, NormativeProcessor

from .trajectories import fractalD, path_length
