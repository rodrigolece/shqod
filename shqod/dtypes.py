"""Common dtypes."""

from typing import List, Iterable, Tuple
import numpy as np

Path = Iterable[np.array]
LexPath = Iterable[int]  # TODO: get rid of this

BoxCounts = Tuple[List[int], List[int]]
