from typing import Union, Optional
from types import ModuleType
import sak
import sak.signal
import warnings
import numpy as np

def mixup(x1: np.ndarray, x2: np.ndarray, alpha: float = 1.0, beta: float = 1.0, axis = None, shuffle: bool = True, rng: Union[np.random.RandomState, None] = None):
    """Adapted from original authors of paper "[1710.09412] mixup: Beyond Empirical Risk Minimization"
    GitHub: https://github.com/facebookresearch/mixup-cifar10/
    """

    # Compute lambda. If hyperparams are incorrect, your loss
    if rng is None:
        rng = np.random
    lmbda = rng.beta(alpha, beta)

    if axis is None:
        axis = 0
        shuffle = False # The default indicates that no batch is used

    # Swap axes to generalize for n-dimensional tensor
    x1 = np.swapaxes(x1,axis,0) # Compatible with pytorch
    x2 = np.swapaxes(x2,axis,0) # Compatible with pytorch

    # Permutation along data axis (allowing batch mixup)
    if shuffle:
        index = rng.permutation(np.arange(x2.shape[0])) # Compatible with pytorch

        # Mix datapoints. If shapes are incompatible, your loss
        xhat = lmbda * x1 + (1 - lmbda) * x2[index, :]
    else:
        # Mix datapoints. If shapes are incompatible, your loss
        xhat = lmbda * x1 + (1 - lmbda) * x2
    
    # Swap axes back
    xhat = np.swapaxes(xhat,axis,0) # Compatible with pytorch

    # Return mixed point and lambda. Left label computation to be project-specific
    return xhat, lmbda
