from typing import Iterable
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import scipy.linalg
import scipy.interpolate

def interpolate_1d_vector(vector: np.ndarray, factor: int) -> np.ndarray:
    """
    Interpolate, i.e. upsample, a given 1D vector by a specific interpolation factor.
    :param vector: 1D data vector
    :param factor: factor for interpolation (must be integer)
    :return: interpolated 1D vector by a given factor
    """
    x = np.arange(np.size(vector))
    y = vector
    f = sp.interpolate.interp1d(x, y)

    x_extended_by_factor = np.linspace(x[0], x[-1], np.size(x) * factor)
    y_interpolated = np.zeros(np.size(x_extended_by_factor))

    i = 0
    for x in x_extended_by_factor:
        y_interpolated[i] = f(x)
        i += 1

    return y_interpolated



def interp(idata: np.ndarray, r: int, l: int = 4, cutoff: float = .5) -> np.ndarray:
    """INTERP Resample data at a higher rate using lowpass interpolation.
    Y = INTERP(X,R) resamples the sequence in vector X at R times
    the original sample rate.  The resulting resampled vector Y is
    R times longer, LENGTH(Y) = R*LENGTH(X).

    A symmetric filter, B, allows the original data to pass through
    unchanged and interpolates between so that the mean square error
    between them and their ideal values is minimized.
    Y = INTERP(X,R,L,CUTOFF) allows specification of arguments
    L and CUTOFF which otherwise default to 4 and .5 respectively.
    2*L is the number of original sample values used to perform the
    interpolation.  Ideally L should be less than or equal to 10.
    The length of B is 2*L*R+1. The signal is assumed to be band
    limited with cutoff frequency 0 < CUTOFF <= 1.0. 
    [Y,B] = INTERP(X,R,L,CUTOFF) returns the coefficients of the
    interpolation filter B.  

    # Example:
    #   Interpolate a signal by a factor of four.

    t = 0:0.001:.029;                       # Time vector
    x = np.sin(2*np.pi*30*t) + np.sin(2*np.pi*60*t);    # Original Signal
    y = interp(x,4);                        # Interpolated Signal
    subplot(211);
    stem(x);
    title('Original Signal');
    subplot(212);
    stem(y); 
    title('Interpolated Signal');

    See also DECIMATE, RESAMPLE, UPFIRDN.

    Author(s): L. Shure, 5-14-87
                 L. Shure, 6-1-88, 12-15-88, revised
    Copyright 1988-2014 The MathWorks, Inc.

    References:
    "Programs for Digital Signal Processing", IEEE Press
    John Wiley & Sons, 1979, Chap. 8.1."""

    if type(idata) == list:
        idata          = np.asarray(idata).reshape((len(idata),1))

    if idata.ndim == 1:
        idata          = idata.reshape((idata.size,1))

    if (l < 1) | (r < 1) | (cutoff <= 0) | (cutoff > 1):
        raise ValueError("(l < 1) | (r < 1) | (cutoff <= 0) | (cutoff > 1)")

    if abs(r-np.fix(r)) > np.finfo(float).eps:
        raise ValueError("r must be an integer")

    if 2*l+1 > len(idata):
        raise ValueError("Invalid dimensions")

    # ALL occurrences of np.sin()/() are using the sinc function for the
    # autocorrelation for the input data. They should all be changed
    # consistently if they are changed at all.
        
    # calculate AP and AM matrices for inversion
    s1                     = sp.linalg.toeplitz(np.asarray(list(range(l)),dtype=float)) + np.finfo(float).eps
    s2                     = sp.linalg.hankel(np.asarray(list(range(2*l-1,l-1,-1)),dtype=float))
    s2p                    = sp.linalg.hankel(np.asarray(list(range(1,l)) + [0],dtype=float))
    s2                     = s2 + np.finfo(float).eps + s2p[l-1::-1,l-1::-1]
    s1                     = np.sin(cutoff*np.pi*s1)/(cutoff*np.pi*s1)
    s2                     = np.sin(cutoff*np.pi*s2)/(cutoff*np.pi*s2)
    ap                     = s1 + s2
    am                     = s1 - s2

    # # Compute matrix inverses using Cholesky decomposition for more robustness
    U                      = sp.linalg.cholesky(ap)
    ap                     = np.matmul(sp.linalg.inv(U),(sp.linalg.inv(U)).T)
    U                      = sp.linalg.cholesky(am)
    am                     = np.matmul(sp.linalg.inv(U),(sp.linalg.inv(U).T))

    # now calculate D based on INV(AM) and INV(AP)
    d                      = np.zeros((2*l,l))
    d[:2*l:2,:]            = ap + am
    d[1:2*l:2,:]           = ap - am

    # set up arrays to calculate interpolating filter B
    x                      = (np.asarray(range(0,r),dtype=float)/r).reshape((1,r))
    y                      = np.zeros((2*l,1))
    y[:2*l:2]              = np.asarray(range(l,0,-1)).reshape((l,1))
    y[1:2*(l+1):2]         = np.asarray(range(l-1,-1,-1)).reshape((l,1))
    X                      = np.ones((2*l,1))
    X[:2*l:2]              = -np.ones((l,1))
    XX                     = np.finfo(float).eps + np.matmul(y,np.ones((1,r))) + np.matmul(X,x)
    y                      = X + y + np.finfo(float).eps
    h                      = np.matmul(.5*(d.T),(np.sin(np.pi*cutoff*XX)/(cutoff*np.pi*XX)))
    b                      = np.zeros((2*l*r+1,1))
    b[:(l*r)]              = h.ravel().reshape((l*r,1))
    b[l*r]                 = np.matmul(.5*(d[:,l-1].T),(np.sin(np.pi*cutoff*y)/(np.pi*cutoff*y)))
    b[l*r+1:(2*l*r+2)]     = b[l*r-1::-1]

    # use the filter B to perform the interpolation
    # if 
    (m,n)                  = np.asarray(idata).shape
    nn                     = max(idata.shape)

    if nn == m:
        odata              = np.zeros((r*nn,1))
    else:
        odata              = np.zeros((1,r*nn))

    odata[:nn*r:r]         = idata

    # Filter a fabricated section of data first (match initial values and first derivatives by
    # rotating the first data points by 180 degrees) to get guess of good initial conditions
    # Filter length is 2*l*r+1 so need that many points; can't duplicate first point or
    # guarantee a zero slope at beginning of sequence
    od                     = np.zeros((2*l*r,1))
    od[:2*l*r:r]           = 2*idata[0] - idata[(2*l):0:-1]
    (od,zi)                = sp.signal.lfilter(b.ravel(),1,od.ravel(),zi=np.zeros(od.shape).ravel())
    (odata,zf)             = sp.signal.lfilter(b.ravel(),1,odata.ravel(),zi=zi.ravel())
    odata[:(nn-l)*r]       = odata[(l*r):(nn*r+1)]

    # make sure right hand points of data have been correctly interpolated and get rid of
    # transients by again matching end values and derivatives of the original data
    if nn == m:
        od                 = np.zeros((2*l*r,1))
    else:
        od                 = np.zeros((1,2*l*r))

    od[:(2*l)*r:r]         = 2*idata[nn-1]-idata[(nn-2):(nn-2*l-2):-1]
    (od,zf)                = sp.signal.lfilter(b.ravel(),1,od.ravel(),zi=zf.ravel())
    odata[nn*r-l*r:nn*r+1] = od[:l*r]

    return (odata,b)


def interp1d(y: np.ndarray, new_size: int, **kwargs) -> np.ndarray:
    if isinstance(new_size, np.ndarray) or isinstance(new_size, pd.Series):
        new_size = new_size.size
    elif isinstance(new_size, list):
        new_size = len(new_size)

    axis = kwargs.get('axis', -1)
    # x axis
    x_from = np.linspace(0,1,y.shape[axis])
    x_to = np.linspace(0,1,new_size)

    return sp.interpolate.interp1d(x_from,y,**kwargs)(x_to)
