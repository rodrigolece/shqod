"""shqod - Origin-Destination tools to work with the SHQ dataset."""

from .io import (
    TidyLoader,
    UntidyLoader,
    read_path_csv,
    read_path_feather,
    read_level_grid,
    read_level_size,
    read_level_flags,
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
