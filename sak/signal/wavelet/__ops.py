from typing import Any, List, Tuple
import numpy as np
from scipy.signal import lfilter
from sak.signal.interpolate import interp
from sak.signal.interpolate import interp1d
from sak.signal import zero_crossings

def wav5t(X: np.ndarray, q1: list, q2: list, q3: list, q4: list, q5: list, mode: str = 'edge') -> np.ndarray:
    # Calculates the wavelet transform using quadratic spline wavelet.
    # It calculates n scales.
    # Author:      Juan Pablo Martinez Cortes
    # Last update: Guillermo Jimenez-Perez 09.02.2019 

    # Compute filter lengths
    l1=len(q1); d1=int(np.floor((l1-1.)/2.))
    l2=len(q2); d2=int(np.floor((l2-1.)/2.))
    l3=len(q3); d3=int(np.floor((l3-1.)/2.))
    l4=len(q4); d4=int(np.floor((l4-1.)/2.))
    l5=len(q5); d5=int(np.floor((l5-1.)/2.))

    # Add dimension if single-lead data
    if X.ndim == 1:
        X = X[:,np.newaxis]

    Wavelet = np.empty((X.shape[0], X.shape[1], 5))

    # np.Pad signal with specific np.padding mode 
    X = np.pad(X,((l5,d5),(0,0)),mode=mode)

    # Implementation as a convolution in the temporal domain
    # filter rather than conv, so as to ther are only "good" samples
    Wavelet[...,0] = lfilter(q1,1,X.T)[:,(l5+d1):(d1-d5)].T
    Wavelet[...,1] = lfilter(q2,1,X.T)[:,(l5+d2):(d2-d5)].T
    Wavelet[...,2] = lfilter(q3,1,X.T)[:,(l5+d3):(d3-d5)].T
    Wavelet[...,3] = lfilter(q4,1,X.T)[:,(l5+d4):(d4-d5)].T
    Wavelet[...,4] = lfilter(q5,1,X.T)[:,(l5+d5):].T

    return Wavelet


def qspfilt5(fs: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """This function obtains the quadratic splines wavelet filterbank filters from
    scale 1 to 4 as a function of the sampling frequency, in order to use filters
    with similar analog frequency behaviour for diferent sampling frecuencies.
    Refer to https://doi.org/10.1109/TBME.2003.821031 for details.

    Author:                 Juan Pablo Martinez Cortes
    Python implementation:  Guillermo Jimenez Perez (24/04/18)
    """

    g               = []
    h               = []
    q               = []
    l               = []
    convolutions    = []

    for i in range(5):
        ZeroList = np.zeros((2**i-1,),dtype=int).tolist()
        g.append(np.asarray([2] + ZeroList + [-2] + ZeroList))
        h.append(0.125*np.asarray([1] + ZeroList + [3] + ZeroList + [3] + ZeroList + [1] + ZeroList))

        if i == 0:
            convolutions.append(np.convolve(1,1))
        else:
            convolutions.append(np.convolve(convolutions[i - 1], h[i - 1]))

        q.append(np.convolve(convolutions[i], g[i]))
        l.append(max(np.argwhere(q[i]).tolist())[0] + 1)
        q[i] = q[i][:l[i]]

    if fs != 250.:
        if fs == 200.:
            q1 = 1.1*np.asarray([5./4, -5./4], dtype=float)
            (q2,_) = interp(np.concatenate(([0],q[1],[0])),4,3,0.4)
            (q3,_) = interp(np.concatenate(([0],q[2],[0])),4,7,0.4)
            (q4,_) = interp(np.concatenate(([0],q[3],[0])),4,7,0.4)
            (q5,_) = interp(np.concatenate(([0],q[4],[0])),4,7,0.4)
            q2 = q2[4:-4:5]
            q3 = q3[5:-4:5]
            q4 = q4[2:-4:5]
            q5 = q5[2:-4:5]
        elif fs == 360.:
            (q1,_) = interp(np.concatenate(([0],q[0],[0])),6,1)
            (q2,_) = interp(np.concatenate(([0],q[1],[0])),6,3)
            (q3,_) = interp(np.concatenate(([0],q[2],[0])),6,7)
            (q4,_) = interp(np.concatenate(([0],q[3],[0])),6,7)
            (q5,_) = interp(np.concatenate(([0],q[4],[0])),6,7)
            q1 = q1[4:-6:5]
            q2 = q2[1:-6:5]
            q3 = q3[5:-6:5]
            q4 = q4[3:-6:5]
            q5 = q5[3:-6:5]
            (q1,_) = interp(np.concatenate(([0],q1,[0])),6,1)
            (q2,_) = interp(np.concatenate(([0],q2,[0])),6,3)
            (q3,_) = interp(np.concatenate(([0],q3,[0])),6,7)
            (q4,_) = interp(np.concatenate(([0],q4,[0])),6,7)
            (q5,_) = interp(np.concatenate(([0],q5,[0])),6,7)
            q1 = q1[2:-6:5]
            q2 = q2[5:-6:5]
            q3 = q3[4:-6:5]
            q4 = q4[4:-6:5]
            q5 = q5[4:-6:5]
        elif fs == 500.:
            (q1,_) = interp(np.concatenate(([0],q[0],[0])),2,1)
            (q2,_) = interp(np.concatenate(([0],q[1],[0])),2,3)
            (q3,_) = interp(np.concatenate(([0],q[2],[0])),2,7)
            (q4,_) = interp(np.concatenate(([0],q[3],[0])),2,7)
            (q5,_) = interp(np.concatenate(([0],q[4],[0])),2,7)
            q1 = q1[1:-2]
            q2 = q2[1:-2]
            q3 = q3[1:-2]
            q4 = q4[1:-2]
            q5 = q5[1:-2]
        elif fs == 1000.:
            (q1,_) = interp(np.concatenate(([0],q[0],[0])),4,1)
            (q2,_) = interp(np.concatenate(([0],q[1],[0])),4,3)
            (q3,_) = interp(np.concatenate(([0],q[2],[0])),4,7)
            (q4,_) = interp(np.concatenate(([0],q[3],[0])),4,7)
            (q5,_) = interp(np.concatenate(([0],q[4],[0])),4,7)
            q1 = q1[1:-4]
            q2 = q2[1:-4]
            q3 = q3[1:-4]
            q4 = q4[1:-4]
            q5 = q5[1:-4]
        else:
            raise NotImplementedError("Frequencies outside [200,250,360,500,1000] Hz not supported.")
    else:
        (q1,q2,q3,q4,q5) = q
            
    return q1,q2,q3,q4,q5


def transform(X: np.ndarray, fs: float, mode: str = 'edge') -> np.ndarray:
    return wav5t(X, *qspfilt5(fs),mode=mode)


def compute_crossings(X: np.ndarray, Wavelet: np.ndarray) -> List[np.ndarray]:
    # Check input
    if (X.ndim != 2) or (Wavelet.ndim != 2):
        raise ValueError("Function supposed to work with wavelets of all leads for a single scale")

    # Compute zero crossings
    crossings = [np.concatenate(([0],zero_crossings(Wavelet[:,lead]),[Wavelet.shape[0]-1])) for lead in range(Wavelet.shape[1])]
    
    # Compute areas for these zero crossings
    areas = [np.array([np.trapz(Wavelet[crossings[lead][c]:crossings[lead][c+1],lead]) for c in range(len(crossings[lead])-1)]) for lead in range(Wavelet.shape[1])]

    
    for lead in range(Wavelet.shape[1]):
        conv_length = len(crossings[lead])-1

        while conv_length > 1:
            mark_incomplete = False

            for location in range(areas[lead].size-conv_length):
                area_left = areas[lead][location]
                area_right = areas[lead][location+conv_length]
                
                areas_between = areas[lead][location+1:location+conv_length]

                filter_left = np.abs(areas_between) >= np.abs(area_left)
                filter_right = np.abs(areas_between) >= np.abs(area_right)

                if not np.any(filter_left | filter_right):
                    areas_to_fuse = np.arange(location+1,location+conv_length)
                    crossings_to_fuse =  np.unique([index_crossing for index_area in areas_to_fuse for index_crossing in range(index_area,index_area+2)])

                    if np.sign(area_left) != np.sign(area_right):
                        if np.sign(area_left) == -1:
                            crossings_to_fuse = np.delete(crossings_to_fuse,X[crossings[lead][crossings_to_fuse],lead].argmin())
                        else:
                            crossings_to_fuse = np.delete(crossings_to_fuse,X[crossings[lead][crossings_to_fuse],lead].argmax())

                    crossings[lead] = np.delete(crossings[lead],crossings_to_fuse)
                    areas[lead] = np.array([np.trapz(Wavelet[crossings[lead][c]:crossings[lead][c+1],lead]) for c in range(len(crossings[lead])-1)])

                    mark_incomplete = True
                    break

            if not mark_incomplete:
                conv_length -= 1
    
    return crossings


def compute_waves(W: np.ndarray, crossings: np.ndarray or list, normalize: bool = True) -> np.ndarray:
    # Step 0: check input
    if W.ndim != 1:
        raise ValueError("Only single wavelet for single lead accepted")
    if isinstance(crossings[0], list):
        raise ValueError("List of crossings per lead not accepted")
        
    # Step 1: compute areas
    areas = np.array([np.trapz(W[crossings[c]:crossings[c+1]]) for c in range(len(crossings)-1)])

    # Shortcut in case of monotonically increasing functions
    if len(crossings) == 2:
        return np.array([np.trapz(W)/np.abs(np.trapz(W))])

    # Allocate output vector
    waves = np.zeros((len(crossings)-2,))

    # Step 2: Iterate over waves
    for c in range(1,len(crossings)-1):
        # 2.1. Select left and right areas
        area_left = areas[c-1]
        area_right = areas[c]

        # 2.2. Consider areas further to the right to account for overshooting
        other_areas = 0
        break_areas = False
        for j in range(c+1, len(areas)):
            other_areas += areas[j]

            # If any wave to the right is higher than the considered right wave, break
            if np.abs(other_areas) > np.abs(area_right):
                other_areas = 0
                break_areas = True
                break
    
        # If any wave on the right is larger than the current, only consider left area
        if break_areas:
            area_right = -area_left
        else:
            if np.abs(area_left) > np.abs(area_right):
                # Account for the unallocated space in the right area
                area_right = area_right + other_areas
            else:
                # Divide remaining space between current and next waves
                eps = np.finfo('float').eps
                normalization_factor = (np.abs(area_left) + eps)/(np.abs(area_left + other_areas) + eps)
                area_right = -area_left + (area_right + other_areas + area_left)*normalization_factor

        # Compute wave
        waves[c-1] = area_left - area_right

        # Update waves with allocated parts of the areas
        areas[c-1] -= area_left
        areas[c]   -= area_right

    if normalize:
        waves = waves/np.max(np.abs(waves))

    return waves


def find_peaks(x: np.ndarray, fs: float = 1000., scale: [1,5] = 5, upscale: int = 5):
    if (scale < 1) or (scale > 5):
        raise ValueError("Scale parameter must be valued in the range [1,5]")
    if x.ndim > 1:
        raise NotImplementedError("Will do in the future")
    x = interp1d(np.copy(x).astype(float),x.size*upscale)
    
    # Obtain wavelet transform of upsampled envelope
    wavelet = transform(x, fs)
    
    # Return wavelet transform to original sampling frequency
    wavelet = interp1d(wavelet,x.size//upscale,axis=0)

    # Isolate nth scale
    wavelet_scale = wavelet[:,0,scale-1]

    # Compute crossings, maxima, minima
    crossings = zero_crossings(wavelet_scale)
    
    return crossings
