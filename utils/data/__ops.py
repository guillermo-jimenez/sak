from typing import Any
from typing import List
from typing import Tuple
from typing import Callable
import numpy as np

class Struct:
    def __init__(self, structure: dict = {}, **kwargs):
        structure.update(kwargs)
        for key, value in structure.items():
            setattr(self, key, value)

def ball_scaling(self, X: np.ndarray, radius: float = 1.0):
    """Balls of radius != 1 not implemented yet"""
    if radius != 1.0:
        raise NotImplementedError("Balls of radius != 1 not implemented yet")
    return X/(np.max(X)-np.min(X)+np.finfo(X.dtype).eps)

