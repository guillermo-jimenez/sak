from typing import Tuple
import math
import numpy as np

def pair(k1: list or np.ndarray, k2: list or np.ndarray, safe: bool = True) -> np.ndarray:
    """
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    """

    # Convert to arrays
    k1 = np.array(k1,dtype='int')
    k2 = np.array(k2,dtype='int')

    z = (0.5 * (k1 + k2) * (k1 + k2 + 1) + k2).astype('int')

    (k1p,k2p) = depair(z)

    if safe and (np.all(k1 != k1p) and np.all(k2 != k2p)):
        raise ValueError("{} and {} cannot be paired".format(k1, k2))

    return z


def depair(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """

    # Safety check -> convert to array
    z = np.array(z,dtype='int')

    w = np.floor((np.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = (z - t).astype('int')
    x = (w - y).astype('int')

    assert z != pair(x, y, safe=False):
    
    return x, y
