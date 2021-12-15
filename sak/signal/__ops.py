from typing import List, Tuple, Iterable, Callable
import numpy as np
import scipy as sp
import scipy.fftpack
from skimage.util import view_as_windows

StandardHeader = np.array(['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

def moving_lambda(x: Iterable, stride: int, lmbda: Callable, axis: int = 0) -> List[Iterable]:
    x = np.swapaxes(np.copy(x),0,axis)
    return np.swapaxes([lmbda(x[i:i+stride]) for i in range(0,len(x),stride)],0,axis)

def sigmoid(x: float or Iterable) -> float or np.ndarray:
    return 1/(1 + np.exp(-x))

def normal(num: int, sigma: float = 1) -> np.ndarray:
    return np.exp(-(np.linspace(-3*sigma,3*sigma,num)**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def power(x: np.ndarray, axis=None) -> float:
    """Compute signal power"""
    return np.mean((x - np.median(x,axis=axis))**2,axis=axis)

def pulse_train(n: int, width: int, offset: int = 0, amplitude: float = 1.0) -> np.ndarray:
    """Compute signal power"""
    return amplitude*(np.arange(offset,n+offset)%(2*width) < width)

def get_mask_boundary(binary_mask: np.ndarray, axis=-1, aslist=True) -> Tuple[list,list]:
    binary_mask = binary_mask.astype(int)
    diff = np.diff(np.pad(binary_mask,((1,1),),'constant',constant_values=0),axis=axis)

    onsets = (np.where(diff ==  1.)[0])
    offsets = (np.where(diff == -1.)[0] - 1)

    if aslist:
        onsets = onsets.tolist()
        offsets = offsets.tolist()
    
    return onsets, offsets

def flatten_along_axis(X: np.ndarray, axis: int = None):
    X = np.copy(X)
    if axis != None:
        for i in reversed(range(axis)):
            X = np.swapaxes(X,i,i+1)
        X = X.reshape((X.shape[0], X.size//X.shape[0]))
    else:
        X = X.flatten()[np.newaxis,:]
    return X

def unflatten_along_axis(X: np.ndarray, shape: Tuple[int], axis: int):
    """Unpacks an array with the same order of operations as 'flatten_along_axis'. 
    Only to be used with that function"""
    X = np.copy(X)
    shape = np.copy(shape).tolist()
    
    # Assert shapes are compatible
    assert X.size == np.prod(shape), "The shapes do not coincide"
    
    # Check position of dimension 0 of flattened array
    for i in reversed(range(axis)):
        # Swap values in shape vector
        val_curr = shape[i]
        val_next = shape[i+1]

        shape[i] = val_next
        shape[i+1] = val_curr
        
    # Reshape array
    X = X.reshape(shape)
    
    # Swap axes to original shape
    for i in range(axis):
        X = np.swapaxes(X,i,i+1)
    return X

def moving_average(x, w=5, axis=-1, **kwargs):
    kwargs["mode"] = kwargs.get("mode","edge")
    x = np.copy(x)
    transpose,squeeze = False,False
    if x.ndim == 1:
        squeeze = True
        x = x[None,:]
    if x.ndim != 2:
        raise ValueError("Function works with 2-dimensional data at most")
    if (axis == 0) or (axis == -2):
        transpose = True
        x = x.T

    # Pad array
    out = np.zeros(x.shape)
    x = np.pad(x,((0,0),(w+1, w+1)), **kwargs)

    # Convolve each element separately
    for i,slice in enumerate(x):
        out[i,] = (np.convolve(slice, np.ones(w), 'same') / w)[w+1:-(w+1)]

    if transpose:
        out = out.T

    if squeeze:
        out = out.squeeze()

    return out

def amplitude(X: np.ndarray, **kwargs) -> np.ndarray:
    return np.max(X,**kwargs) - np.min(X,**kwargs)
    
def abs_max(X: np.ndarray, **kwargs) -> np.ndarray:
    return np.max(np.abs(X),**kwargs)
    
def min_max_ratio(X: np.ndarray, **kwargs) -> np.ndarray:
    maximum = np.max(X, **kwargs)
    minimum = np.min(X, **kwargs)
    return 1-np.min([np.abs(maximum),np.abs(minimum)])/np.max([np.abs(maximum),np.abs(minimum)])
    
def on_off_correction(X: np.ndarray) -> np.ndarray:
    output = np.copy(X)
    
    if output.ndim == 1:
        output = output[:,None]

    # Compute onsets and offsets
    onsets = output[0,:]
    offsets = output[-1,:]

    # Correct inputs
    output += np.linspace(-onsets,-offsets,output.shape[0])

    # Squeeze if necessary
    if output.ndim != X.ndim:
        return output.squeeze(axis=tuple(range(1,output.ndim)))
    else:
        return output
    
def abs_max_is_positive(X: np.ndarray, axis: int = None) -> np.ndarray:
    X = flatten_along_axis(X, axis)
    return X[np.arange(X.shape[0]),np.abs(X).argmax(axis=-1)] == X[np.arange(X.shape[0]),X.argmax(axis=-1)].squeeze()

def signed_maxima(X: np.ndarray, axis: int = None) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        input_type = type(X)
        X = np.array(X)
        if X.dtype == 'O':
            raise ValueError("Invalid input data type: {}".format(input_type))
    X = flatten_along_axis(X, axis)
    return X[np.arange(X.shape[0]),np.abs(X).argmax(axis=-1)].squeeze()

def signed_minima(X: np.ndarray, axis: int = None) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        input_type = type(X)
        X = np.array(X)
        if X.dtype == 'O':
            raise ValueError("Invalid input data type: {}".format(input_type))
    X = flatten_along_axis(X, axis)
    return X[np.arange(X.shape[0]),np.abs(X).argmin(axis=-1)].squeeze()

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
    if X.ndim == 1:
        return sampfrom + np.where(np.diff(np.sign(np.diff(X)),prepend=0,append=0).T < -1)[0]
    elif X.ndim == 2:
        return [sampfrom + np.where(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T < -1)[0] for j in range(X.shape[1])]
    else:
        raise NotImplementedError("maxima function not implemented for arrays larger than 2D")

def minima(X: np.ndarray, sampfrom: int = 0, sampto: int = None) -> List[np.ndarray]:
    """Returns relative minima of array"""
    # X = ordering_N_lead(X)
    X = X[sampfrom:sampto,]
    if X.ndim == 1:
        return sampfrom + np.where(np.diff(np.sign(np.diff(X)),prepend=0,append=0).T > 1)[0]
    elif X.ndim == 2:
        return [sampfrom + np.where(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T > 1)[0] for j in range(X.shape[1])]
    else:
        raise NotImplementedError("minima function not implemented for arrays larger than 2D")

def extrema(X: np.ndarray, sampfrom: int = 0, sampto: int = None) -> List[np.ndarray]:
    """Returns relative minima of array"""
    # X = ordering_N_lead(X)
    X = X[sampfrom:sampto,]
    if X.ndim == 1:
        return sampfrom + np.where(np.abs(np.diff(np.sign(np.diff(X)),prepend=0,append=0).T) > 1)[0]
    elif X.ndim == 2:
        return [sampfrom + np.where(np.abs(np.diff(np.sign(np.diff(X[:,j])),prepend=0,append=0).T) > 1)[0] for j in range(X.shape[1])]
    else:
        raise NotImplementedError("extrema function not implemented for arrays larger than 2D")

def zero_crossings(X: np.ndarray, axis: int = None, inclusive: bool = False) -> List[np.ndarray]:
    """Returns zero crossings of array"""
    # X = ordering_N_lead(X)
    if X.ndim == 1:
        signs = np.sign(X)
        zero_diff = (signs == 0)*(np.diff(signs,prepend=signs[0]) + np.diff(signs,append=signs[-1]))
        nonzero_diff = (np.abs(np.diff(signs,prepend=signs[0])) == 2)

        diff = zero_diff + nonzero_diff
        crossings = np.where(diff)[0]
        if inclusive:
            crossings = np.concatenate(([0],crossings,[X.shape[0]]))
        return crossings
    elif X.ndim == 2:
        crossings = [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])))[0] for j in range(X.shape[1])]
        if inclusive:
            crossings = [np.concatenate(([0],c,[X.shape[0]])) for c in crossings]
        return crossings
    else:
        raise NotImplementedError("zero_crossings function not implemented for arrays larger than 2D")

def positive_zero_crossings(X: np.ndarray) -> List[np.ndarray]:
    """Returns zero crossings of array"""
    # X = ordering_N_lead(X)
    if X.ndim == 1:
        return np.where(np.diff(np.sign(X),prepend=np.sign(X[0])) < 0)[0]
    elif X.ndim == 2:
        return [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])) < 0)[0] for j in range(X.shape[1])]
    else:
        raise NotImplementedError("positive_zero_crossings function not implemented for arrays larger than 2D")

def negative_zero_crossings(X: np.ndarray) -> List[np.ndarray]:
    """Returns zero crossings of array"""
    # X = ordering_N_lead(X)
    if X.ndim == 1:
        return np.where(np.diff(np.sign(X),prepend=np.sign(X[0])) > 0)[0]
    elif X.ndim == 2:
        return [np.where(np.diff(np.sign(X[:,j]),prepend=np.sign(X[0,j])) > 0)[0] for j in range(X.shape[1])]
    else:
        raise NotImplementedError("negative_zero_crossings function not implemented for arrays larger than 2D")


def zero_crossing_areas(X: np.ndarray, axis: int = None, normalize: bool = False) -> List[np.ndarray]:
    """Returns zero crossings of array"""
    # X = ordering_N_lead(X)
    eps = np.finfo(X.dtype).eps
    if X.ndim == 1:
        crossings = zero_crossings(X)
        if 0 not in crossings:
            crossings = np.concatenate(([0],crossings))
        if X.size-1 not in crossings:
            crossings = np.concatenate((crossings,[X.size-1]))

        if normalize:
            areas = np.array([np.trapz(X[a:b])/(b-a+eps) for a,b in zip(crossings[:-1],crossings[1:])])
        else:
            areas = np.array([np.trapz(X[a:b]) for a,b in zip(crossings[:-1],crossings[1:])])
    elif (X.ndim == 2) and axis in [0,1]:
        crossings = zero_crossings(X, axis=axis)
        if 0 not in crossings:
            crossings = np.concatenate(([0],crossings))
        if X.size-1 not in crossings:
            crossings = np.concatenate((crossings,[X.size-1]))

        if axis == 0:
            if normalize:
                areas = [np.array([np.trapz(X[a:b,i])/(b-a+eps)  for a,b in zip(crossings[:-1],crossings[1:])]) for i in range(X.shape[1])]
            else:
                areas = [np.array([np.trapz(X[a:b,i]) for a,b in zip(crossings[:-1],crossings[1:])]) for i in range(X.shape[1])]
        else:
            if normalize:
                areas = [np.array([np.trapz(X[i,a:b])/(b-a+eps)  for a,b in zip(crossings[:-1],crossings[1:])]) for i in range(X.shape[0])]
            else:
                areas = [np.array([np.trapz(X[i,a:b]) for a,b in zip(crossings[:-1],crossings[1:])]) for i in range(X.shape[0])]
    else:
        raise NotImplementedError("zero_crossings function not implemented for arrays larger than 2D")

    return areas


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
    assert x.shape[0] == y.shape[0], "The size of x and y must be the same"

    # Retrieve Nx
    Nx = x.shape[0]
    
    # Define maxlags
    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 0:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    # Correlate signals
    corr_mode = 'full' if maxlags != 0 else 'valid'
    c = np.correlate(x, y, mode=corr_mode)

    # Proceed to normalize output
    if normed:
        # this is the transformation function
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) + max([np.finfo(x.dtype).eps,np.finfo(y.dtype).eps])
        c = np.true_divide(c,n)

    # Retrieve lags vector and refine if maxlags < Nx-1
    lags = np.arange(-maxlags, maxlags + 1)
    if maxlags > 0:
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
    if maxlags is None:
        maxlags = s-1
    if maxlags >= s or maxlags < 0:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % s)

    # Pad input
    matrix_padded = np.pad(matrix,((0,0),(s,s)),constant_values=0)
    matrix_padded = view_as_windows(matrix_padded,(1,s,)).squeeze()

    lags = np.arange(-s,s+1)
    if (maxlags < matrix.shape[-1]) and (maxlags > 0):
        filt = (lags >= -maxlags) & (lags <= maxlags)
        lags = lags[filt]
        matrix_padded = matrix_padded[:,filt,:]
    if maxlags == 0:
        filt = (lags == 0)
        matrix_padded = matrix_padded[:,filt,:]

    # Elementwise multiplications across all elements in axis=0, 
    # and then summation along axis=1
    out = np.einsum('ijkl,ijkl->ijk',matrix[None,:,None,:],matrix_padded[:,None,:,:])
    if normed:
        norm_x = np.einsum('ijk,ijk->ij',matrix[:,None,:],matrix[:,None,:])
        norm = np.sqrt(norm_x*norm_x.T)+np.finfo(matrix.dtype).eps
        out = np.true_divide(out,norm[...,None])

    # Use valid mask to skip columns and have the final output
    return out, lags

def approx_N_cycles(x):
    """http://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_periodicity_finder.html"""
    # Get estimate of number of waves
    ft_x = sp.fftpack.fft(x, axis=0)[1:]
    frequencies = sp.fftpack.fftfreq(x.shape[0])[1:]
    periods = 1 / frequencies
    cycles = int(np.round(x.size/periods[np.argmax(abs(ft_x))]))
    
    return cycles

