from typing import Tuple, Callable, Iterable, List, Union
import torch
import torch.utils
import torch.utils.data
import numpy as np
import numpy.random
import os

from utils.__ops import required
from utils.__ops import check_required

def power(x: torch.tensor, axis=None) -> torch.tensor:
    if axis is None: return torch.mean((x - torch.mean(x))**2)
    else:            return torch.mean((x - torch.mean(x,axis=axis))**2,axis=axis)
