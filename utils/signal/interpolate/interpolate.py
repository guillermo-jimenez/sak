from numpy import arange
from numpy import size
from numpy import zeros
from numpy import ndarray
from numpy import ones
from numpy import concatenate
from numpy import asarray
from numpy import fix
from numpy import linspace
from numpy import sin
from numpy import matmul
from numpy import finfo
from numpy import pi
from scipy.interpolate import interp1d
from scipy.signal import lfilter
from scipy.linalg import hankel
from scipy.linalg import toeplitz
from scipy.linalg import cholesky
from scipy.linalg import inv

def interpolate_1d_vector(vector: ndarray, factor: int) -> ndarray:
    """
    Interpolate, i.e. upsample, a given 1D vector by a specific interpolation factor.
    :param vector: 1D data vector
    :param factor: factor for interpolation (must be integer)
    :return: interpolated 1D vector by a given factor
    """
    x = arange(size(vector))
    y = vector
    f = interp1d(x, y)

    x_extended_by_factor = linspace(x[0], x[-1], size(x) * factor)
    y_interpolated = zeros(size(x_extended_by_factor))

    i = 0
    for x in x_extended_by_factor:
        y_interpolated[i] = f(x)
        i += 1

    return y_interpolated



def interp(idata: ndarray, r: int, l: int = 4, cutoff: float = .5) -> ndarray:
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
    x = sin(2*pi*30*t) + sin(2*pi*60*t);    # Original Signal
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
        idata          = asarray(idata).reshape((len(idata),1))

    if idata.ndim == 1:
        idata          = idata.reshape((idata.size,1))

    if (l < 1) | (r < 1) | (cutoff <= 0) | (cutoff > 1):
        raise ValueError("(l < 1) | (r < 1) | (cutoff <= 0) | (cutoff > 1)")

    if abs(r-fix(r)) > finfo(float).eps:
        raise ValueError("r must be an integer")

    if 2*l+1 > len(idata):
        raise ValueError("Invalid dimensions")

    # ALL occurrences of sin()/() are using the sinc function for the
    # autocorrelation for the input data. They should all be changed
    # consistently if they are changed at all.
        
    # calculate AP and AM matrices for inversion
    s1                     = toeplitz(asarray(list(range(l)),dtype=float)) + finfo(float).eps
    s2                     = hankel(asarray(list(range(2*l-1,l-1,-1)),dtype=float))
    s2p                    = hankel(asarray(list(range(1,l)) + [0],dtype=float))
    s2                     = s2 + finfo(float).eps + s2p[l-1::-1,l-1::-1]
    s1                     = sin(cutoff*pi*s1)/(cutoff*pi*s1)
    s2                     = sin(cutoff*pi*s2)/(cutoff*pi*s2)
    ap                     = s1 + s2
    am                     = s1 - s2

    # # Compute matrix inverses using Cholesky decomposition for more robustness
    U                      = cholesky(ap)
    ap                     = matmul(inv(U),(inv(U)).T)
    U                      = cholesky(am)
    am                     = matmul(inv(U),(inv(U).T))

    # now calculate D based on INV(AM) and INV(AP)
    d                      = zeros((2*l,l))
    d[:2*l:2,:]            = ap + am
    d[1:2*l:2,:]           = ap - am

    # set up arrays to calculate interpolating filter B
    x                      = (asarray(range(0,r),dtype=float)/r).reshape((1,r))
    y                      = zeros((2*l,1))
    y[:2*l:2]              = asarray(range(l,0,-1)).reshape((l,1))
    y[1:2*(l+1):2]         = asarray(range(l-1,-1,-1)).reshape((l,1))
    X                      = ones((2*l,1))
    X[:2*l:2]              = -ones((l,1))
    XX                     = finfo(float).eps + matmul(y,ones((1,r))) + matmul(X,x)
    y                      = X + y + finfo(float).eps
    h                      = matmul(.5*(d.T),(sin(pi*cutoff*XX)/(cutoff*pi*XX)))
    b                      = zeros((2*l*r+1,1))
    b[:(l*r)]              = h.ravel().reshape((l*r,1))
    b[l*r]                 = matmul(.5*(d[:,l-1].T),(sin(pi*cutoff*y)/(pi*cutoff*y)))
    b[l*r+1:(2*l*r+2)]     = b[l*r-1::-1]

    # use the filter B to perform the interpolation
    # if 
    (m,n)                  = asarray(idata).shape
    nn                     = max(idata.shape)

    if nn == m:
        odata              = zeros((r*nn,1))
    else:
        odata              = zeros((1,r*nn))

    odata[:nn*r:r]         = idata

    # Filter a fabricated section of data first (match initial values and first derivatives by
    # rotating the first data points by 180 degrees) to get guess of good initial conditions
    # Filter length is 2*l*r+1 so need that many points; can't duplicate first point or
    # guarantee a zero slope at beginning of sequence
    od                     = zeros((2*l*r,1))
    od[:2*l*r:r]           = 2*idata[0] - idata[(2*l):0:-1]
    (od,zi)                = lfilter(b.ravel(),1,od.ravel(),zi=zeros(od.shape).ravel())
    (odata,zf)             = lfilter(b.ravel(),1,odata.ravel(),zi=zi.ravel())
    odata[:(nn-l)*r]       = odata[(l*r):(nn*r+1)]

    # make sure right hand points of data have been correctly interpolated and get rid of
    # transients by again matching end values and derivatives of the original data
    if nn == m:
        od                 = zeros((2*l*r,1))
    else:
        od                 = zeros((1,2*l*r))

    od[:(2*l)*r:r]         = 2*idata[nn-1]-idata[(nn-2):(nn-2*l-2):-1]
    (od,zf)                = lfilter(b.ravel(),1,od.ravel(),zi=zf.ravel())
    odata[nn*r-l*r:nn*r+1] = od[:l*r]

    return (odata,b)
