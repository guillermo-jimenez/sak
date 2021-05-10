from typing import Any, List, Tuple, Union, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sak
from matplotlib.figure import Figure


def N_leads(
        x: np.ndarray, 
        y: np.ndarray = None, 
        header: list = None, 
        n: int = 6, 
        samplefrom: int = 0, 
        sampleto: int = None, 
        figsizemultiplier: int = 2, 
        returns: bool = False, 
        **kwargs: dict
    ) -> Tuple[Figure, np.ndarray]:

    # Check input data
    if y is None:
        y = x
        x = np.arange(y.shape[0])
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y vectors inputted with different shapes '{}' and '{}'".format(x.shape[0],y.shape[0]))

    # Set default values
    kwargs['ncols'] = kwargs.get('ncols', int(np.ceil(y.shape[1]/n)))
    kwargs['nrows'] = kwargs.get('nrows', y.shape[1] if kwargs['ncols'] == 1 else n)
    kwargs['figsize'] = kwargs.get('figsize', (figsizemultiplier*kwargs['ncols'],figsizemultiplier*kwargs['nrows']))
        
    # Create figure
    fig,ax = plt.subplots(**kwargs)
    ax = np.array([ax])
    if ax.ndim == 1:
        ax = ax[np.newaxis]
        
    for j in range(kwargs['nrows']):
        for i in range(kwargs['ncols']):
            index = i*n+j
            if index < y.shape[1]:
                if x.ndim > 1:
                    ax[j][i].plot(x[:,index], y[:,index])
                else:
                    ax[j][i].plot(x, y[:,index])
                
    if header is not None:
        fig.legend(header)
    
    if returns:
        return fig,ax


def segmentation(
        x: np.ndarray, 
        mask: np.ndarray, 
        ax: Optional[Union[np.ndarray, matplotlib.axes.Axes]] = None,
        returns: bool = False, 
        fig: Optional[matplotlib.figure.Figure] = None,
        **kwargs: dict
    ) -> Tuple[Figure, np.ndarray]:

    # Manage inputs
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().detach().numpy()
    if not isinstance(x, np.ndarray):
        x = x.cpu().detach().numpy()
    if mask.shape[1] < mask.shape[0]: 
        mask = mask.T

    # Retrieve stuff from mask
    pon,poff = sak.signal.get_mask_boundary(mask[0,:])
    qrson,qrsoff = sak.signal.get_mask_boundary(mask[1,:])
    ton,toff = sak.signal.get_mask_boundary(mask[2,:])

    # Set default values
    kwargs['figsize'] = kwargs.get('figsize', (16,4))

    # Define figure correctly
    if fig is None:
        fig = plt.figure(**kwargs)
    if ax is None:
        ax = fig.gca()

    # Plot the signal information
    ax.plot(x)
    [ax.axvspan(on,off,color='red',alpha=0.15) for on,off in zip(pon,poff)]
    [ax.axvspan(on,off,color='green',alpha=0.15) for on,off in zip(qrson,qrsoff)]
    [ax.axvspan(on,off,color='magenta',alpha=0.15) for on,off in zip(ton,toff)]

    if returns: return fig,ax
 

def standard(
        x: np.ndarray, 
        y: np.ndarray = None, 
        header: list = None, 
        returns: bool = False, 
        **kwargs: dict
    ) -> Tuple[Figure, np.ndarray]:

    # Check input data
    if y is None:
        y = x
        x = np.arange(y.shape[0])
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y vectors inputted with different shapes '{}' and '{}'".format(x.shape[0],y.shape[0]))

    # Set default values
    kwargs['figsize'] = kwargs.get('figsize', (4,12))

    fig,ax = plt.subplots(nrows = 6, ncols = 2, **kwargs)
    [ax[i,0].set_facecolor([0.1,0.8,0,0.1]) for i in range(3)]
    [ax[i,0].set_facecolor([1,0.3,1,0.1]) for i in range(3,6)]
    [ax[i,1].set_facecolor([1,0,0,0.1]) for i in range(6)]
    if x.ndim > 1:
        [ax[i%6][i//6].plot(x[:,i], y[:,i]) for i in range(y.shape[1])]
    else:
        [ax[i%6][i//6].plot(x, y[:,i]) for i in range(y.shape[1])]
        
    if header is not None:
        [ax[i%6][i//6].set_title(header[i]) for i in range(len(header))]

    # Set y ticks to the right in the right column
    for i in range(6):
        ax[i%6][1].yaxis.tick_right()

    # Set limits
    [[ax[i,j].set_xlim([x[0],x[-1]]) for j in range(ax.shape[1])] for i in range(ax.shape[0])]

    # Make the subplots closer, remove x axis ticks (except bottom ones)
    [[ax[i,j].set_xticks([]) for i in range(5)] for j in range(2)]
    fig.subplots_adjust(wspace=0.01,hspace=0.01)

    if returns:
        return fig,ax
    
def wavelets(
        y: np.ndarray, 
        W: np.ndarray, 
        scales: list = None, 
        figsizemultiplier: int = 2, 
        returns: bool = False, 
        fig: Figure = None, 
        ax: np.ndarray = None, 
        **kwargs: dict
    ) -> Tuple[Figure, np.ndarray]:

    # Fix if single-lead
    if y.ndim == 1:
        y = y[:,None]

    # Check if "scales" argument is correctly defined
    if scales is None:
        scales = range(W.shape[-1])
    elif not isinstance(scales,list):
        # Type casting: if not possible, should raise TypeError
        scales = list(scales)

        if not isinstance(scales[0],int):
            raise ValueError("scales argument should be iterable of wavelet scales (int)")

    # Check if wavelet and signal have compatible shapes
    if y.shape[1] != W.shape[1]:
        raise ValueError("y and W vectors inputted with different shapes '{}' and '{}'".format(y.shape[1],W.shape[1]))

    # Set default values
    kwargs['ncols'] = kwargs.get('ncols', y.shape[1])
    kwargs['nrows'] = kwargs.get('nrows', len(scales)+1)
    kwargs['figsize'] = kwargs.get('figsize', (figsizemultiplier*kwargs['ncols'],figsizemultiplier*kwargs['nrows']))

    # Check if figure and axes already inputted into plotter, and if correcly defined
    if not (isinstance(fig,Figure) and isinstance(ax,np.ndarray)):
        fig,ax = plt.subplots(**kwargs)

    # Fix if single-lead
    if ax.ndim == 1:
        ax = ax[:,None]

    # Plot signal
    [ax[0,j].plot(y[:,j]) for j in range(y.shape[1])]
    [[ax[i+1,j].plot(W[:,j,scales[i]]) for j in range(y.shape[1])] for i in range(len(scales))]
    [[ax[i+1,j].plot(np.zeros_like(W[:,j,scales[i]])) for j in range(y.shape[1])] for i in range(len(scales))]
    
    if returns:
        return fig,ax



