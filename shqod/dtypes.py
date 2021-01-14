"""Common dtypes."""

from typing import List, Iterable, Tuple
import numpy as np

Trajec = Iterable[np.array]
LexTrajec = Iterable[int]  # TODO: get rid of this
MatTrajec = np.ndarray

BoxCounts = Tuple[List[int], List[int]]
