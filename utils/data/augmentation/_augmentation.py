import utils
import utils.signal
import numpy as np

def mixup(x1: np.ndarray, x2: np.ndarray, alpha: float = 1.0, beta: float = 1.0, axis = None, shuffle: bool = True):
    """Adapted from original authors of paper "[1710.09412] mixup: Beyond Empirical Risk Minimization"
    GitHub: https://github.com/facebookresearch/mixup-cifar10/
    """

    # Compute lambda. If hyperparams are incorrect, your loss
    lmbda = np.random.beta(alpha, beta)

    if axis is None:
        axis = 0

    # Swap axes to generalize for n-dimensional tensor
    x1 = np.swapaxes(x1,axis,0)
    x2 = np.swapaxes(x2,axis,0)

    # Permutation along data axis (allowing batch mixup)
    if shuffle:
        index = np.random.permutation(np.arange(x2.shape[0]))

        # Mix datapoints. If shapes are incompatible, your loss
        xhat = lmbda * x1 + (1 - lmbda) * x2[index, :]
    else:
        # Mix datapoints. If shapes are incompatible, your loss
        xhat = lmbda * x1 + (1 - lmbda) * x2
    
    # Swap axes back
    xhat = np.swapaxes(xhat,axis,0)

    # Return mixed point and lambda. Left label computation to be project-specific
    return xhat, lmbda
