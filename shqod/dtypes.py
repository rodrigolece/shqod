"""Common dtypes."""

from typing import List, Iterable, Tuple
import numpy as np

Trajec = Iterable[Tuple[int, int]]
LexTrajec = Iterable[int]
MatTrajec = np.ndarray

BoxCounts = Tuple[List[int], List[int]]
