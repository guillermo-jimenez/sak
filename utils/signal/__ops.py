from typing import List, Tuple, Iterable
import numpy as np
from skimage.util import view_as_windows

StandardHeader = np.array(['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

def sigmoid(x: float or Iterable) -> float or np.ndarray:
    return 1/(1 + np.exp(-x))

def get_mask_boundary(binary_mask: np.ndarray, axis=-1) -> Tuple[list,list]:
    binary_mask = binary_mask.astype(int)
    diff = np.diff(np.pad(binary_mask,((1,1),),'constant',constant_values=0),axis=axis)

    onsets = (np.where(diff ==  1.)[0]).tolist()
    offsets = (np.where(diff == -1.)[0] - 1).tolist()
    
    return onsets, offsets

def flatten_along_axis(X: np.ndarray, axis: int = None):
    if axis != None:
        X = X.reshape((X.shape[axis], np.prod(np.delete(X.shape,axis))))
    else:
        X = X.flatten()[np.newaxis,:]
    return X

def amplitude(X: np.ndarray, axis: Tuple = None) -> np.ndarray:
    return np.max(X,axis=axis) - np.min(X,axis=axis)
    
def abs_max(X: np.ndarray, axis: Tuple = None) -> np.ndarray:
    return np.max(np.abs(X),axis=axis)
    
def min_max_ratio(X: np.ndarray, axis: Tuple = None) -> np.ndarray:
    maximum = np.max(X,axis=axis)
    minimum = np.min(X,axis=axis)
    return 1-np.min([np.abs(maximum),np.abs(minimum)])/np.max([np.abs(maximum),np.abs(minimum)])
    
def on_off_correction(X: np.ndarray) -> np.ndarray:
    X = np.copy(X)
    
    if X.ndim == 1:
        X = X[:,None]
    
    for i in range(X.shape[-1]):
        on = X[0,i]
        off = X[-1,i]

        X[:,i] += np.linspace(-on,-off,X.shape[0])
        
    return X.squeeze(axis=tuple(range(1,X.ndim)))
    
def abs_max_is_positive(X: np.ndarray, axis: int = None) -> np.ndarray:
    X = flatten_along_axis(X, axis)
    return X[np.arange(X.shape[0]),np.abs(X).argmax(axis=-1)] == X[np.arange(X.shape[0]),X.argmax(axis=-1)].squeeze()

def signed_maxima(X: np.ndarray, axis: int = None) -> np.ndarray:
    X = flatten_along_axis(X, axis)
    return X[np.arange(X.shape[0]),np.abs(X).argmax(axis=-1)].squeeze()

def is_max(X: np.ndarray, sample: int) -> bool:
    """Returns boolean true if sample is maxima of array"""
    return np.diff(np.sign(np.diff(X)),prepend=0,append=0).T[sample] < -1

def is_min(X: np.ndarray, sample: int) -> bool:
    """Returns boolean true if sample is minima of array"""
    return np.diff(np.sign(np.diff(X)),prepend=0,append=0).T[sample] > 1

def maxima(X: np.ndarray, sampfrom: int = 0, sampto: int = None) -> List[np.ndarray]:
    """Returns relative maxima of array"""
    # X = ordering_N_lead(X)
    X = X[sampfrom:sampto,]
    return [sampfrom + np.where(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T < -1)[0] for j in range(X.shape[1])]

def minima(X: np.ndarray, sampfrom: int = 0, sampto: int = None) -> List[np.ndarray]:
    """Returns relative minima of array"""
    # X = ordering_N_lead(X)
    X = X[sampfrom:sampto,]
    return [sampfrom + np.where(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T > 1)[0] for j in range(X.shape[1])]

def extrema(X: np.ndarray, sampfrom: int = 0, sampto: int = None) -> List[np.ndarray]:
    """Returns relative minima of array"""
    # X = ordering_N_lead(X)
    X = X[sampfrom:sampto,]
    return [sampfrom + np.where(np.abs(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T) > 1)[0] for j in range(X.shape[1])]

def zero_crossings(X: np.ndarray, axis: int = None) -> List[np.ndarray]:
    """Returns zero crossings of array"""
    if X.ndim == 1:
        X = X[:,None]
    # X = ordering_N_lead(X)
    return [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])))[0] for j in range(X.shape[1])]

def positive_zero_crossings(X: np.ndarray) -> List[np.ndarray]:
    """Returns zero crossings of array"""
    # X = ordering_N_lead(X)
    return [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])) < 0)[0] for j in range(X.shape[1])]

def negative_zero_crossings(X: np.ndarray) -> List[np.ndarray]:
    """Returns zero crossings of array"""
    # X = ordering_N_lead(X)
    return [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])) > 0)[0] for j in range(X.shape[1])]

def xcorr(x: np.ndarray, y: np.ndarray = None, normed: bool = True, maxlags: int = None) -> List[np.ndarray]:
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    # Order dimensions of both vectors
    if y is not None:
        if (x.ndim == 2) or (y.ndim == 2):
            return [__xcorr_single(x[:,j],y[:,j],normed,maxlags) for j in range(x.shape[1])]
        else:
            x = x.squeeze()
            y = y.squeeze()
            return __xcorr_single(x,y,normed,maxlags)
    else:
        return __xcorr_matrix(x, normed, maxlags)



def __xcorr_single(x: np.ndarray, y: np.ndarray, normed: bool, maxlags: int) -> Tuple[np.ndarray, np.ndarray]:
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


def __xcorr_matrix(matrix, normed, maxlags):
    # Initial checks
    if matrix.ndim != 2:
        raise ValueError("prepared for 2D inputs")
    if matrix.shape[0] < 1:
        raise ValueError("Minimum of 2 samples needed")
        
    # Retrieve shape
    n,s = matrix.shape

    # Pad input
    matrix_padded = np.pad(matrix,((0,0),(s-1,s-1)),constant_values=0)
    matrix_padded = view_as_windows(matrix_padded,(1,matrix.shape[-1],)).squeeze()

    lags = np.arange(-matrix.shape[-1]+1,matrix.shape[-1])
    if maxlags and (maxlags < matrix.shape[-1]) and (maxlags > 0):
        filt = (lags > -maxlags+1) & (lags < maxlags)
        lags = lags[filt]
        matrix_padded = matrix_padded[:,filt,:]
        
    # Elementwise multiplications across all elements in axis=0, 
    # and then summation along axis=1
    out = np.einsum('ijkl,ijkl->ijk',matrix_padded[None,:,:,:],matrix_padded[:,None,:,:])
    if normed:
        norm_x = np.einsum('ijk,ijk->ij',matrix[:,None,:],matrix[:,None,:])
        norm = np.sqrt(norm_x*norm_x.T)
        out = np.true_divide(out,norm[...,None])

    # Use valid mask to skip columns and have the final output
    return out, lags