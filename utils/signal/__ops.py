from typing import List
from typing import Tuple
import numpy as np

def on_off_correction(X: np.ndarray) -> np.ndarray:
    X = np.copy(X)
    
    if X.ndim == 1:
        X = X[:,None]
    
    for i in range(X.shape[-1]):
        on = X[0,i]
        off = X[-1,i]

        X[:,i] += np.linspace(-on,-off,X.shape[0])
        
    return X.squeeze()
    
def abs_max_is_positive(X: np.ndarray) -> List[np.ndarray]:
    X = ordering_N_lead(X)
    return (X[np.abs(X).argmax(axis=0),np.arange(X.shape[1])] == X[X.argmax(axis=0),np.arange(X.shape[1])]).tolist()

def signed_maxima(X: np.ndarray) -> List[np.ndarray]:
    X = ordering_N_lead(X)
    return X[np.abs(X).argmax(axis=0),np.arange(X.shape[1])].tolist()

def is_max(X: np.ndarray, sample: int) -> bool:
    """Returns boolean true if sample is maxima of tensor"""
    return np.diff(np.sign(np.diff(X)),prepend=0,append=0).T[sample] < -1

def is_min(X: np.ndarray, sample: int) -> bool:
    """Returns boolean true if sample is minima of tensor"""
    return np.diff(np.sign(np.diff(X)),prepend=0,append=0).T[sample] > 1

def maxima(X: np.ndarray, sampfrom: int = 0, sampto: int = None) -> List[np.ndarray]:
    """Returns relative maxima of tensor"""
    X = ordering_N_lead(X)
    X = X[sampfrom:sampto,]
    return [sampfrom + np.where(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T < -1)[0] for j in range(X.shape[1])]

def minima(X: np.ndarray, sampfrom: int = 0, sampto: int = None) -> List[np.ndarray]:
    """Returns relative minima of tensor"""
    X = ordering_N_lead(X)
    X = X[sampfrom:sampto,]
    return [sampfrom + np.where(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T > 1)[0] for j in range(X.shape[1])]

def extrema(X: np.ndarray, sampfrom: int = 0, sampto: int = None) -> List[np.ndarray]:
    """Returns relative minima of tensor"""
    X = ordering_N_lead(X)
    X = X[sampfrom:sampto,]
    return [sampfrom + np.where(np.abs(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T) > 1)[0] for j in range(X.shape[1])]

def zero_crossings(X: np.ndarray) -> List[np.ndarray]:
    """Returns zero crossings of tensor"""
    X = ordering_N_lead(X)
    return [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])))[0] for j in range(X.shape[1])]

def positive_zero_crossings(X: np.ndarray) -> List[np.ndarray]:
    """Returns zero crossings of tensor"""
    X = ordering_N_lead(X)
    return [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])) < 0)[0] for j in range(X.shape[1])]

def negative_zero_crossings(X: np.ndarray) -> List[np.ndarray]:
    """Returns zero crossings of tensor"""
    X = ordering_N_lead(X)
    return [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])) > 0)[0] for j in range(X.shape[1])]

def xcorr(x: np.ndarray, y: np.ndarray, normed: bool = True, maxlags: int = None) -> List[np.ndarray]:
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    # Order dimensions of both vectors
    if (x.ndim == 2) or (y.ndim == 2):
        x =  ordering_N_lead(x)
        y =  ordering_N_lead(y)
        return [__xcorr_aux(x[:,j],y[:,j],normed,maxlags) for j in range(x.shape[1])]
    else:
        x = x.squeeze()
        y = y.squeeze()
        return __xcorr_aux(x,y,normed,maxlags)


def __xcorr_aux(x: np.ndarray, y: np.ndarray, normed: bool, maxlags: int) -> Tuple[np.ndarray, np.ndarray]:
    # Check dimensions
    Nx = x.shape[0]
    if Nx != y.shape[0]:
        raise ValueError('x and y must be equal length')
    
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return c, lags