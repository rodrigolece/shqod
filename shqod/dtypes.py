"""Common dtypes."""

from typing import Iterable, Tuple
import numpy as np

Trajec = Iterable[Tuple[int, int]]
LexTrajec = Iterable[int]
MatTrajec = np.ndarray
