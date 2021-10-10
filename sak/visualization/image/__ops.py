from typing import Any, List, Tuple, Union, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sak
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from sak.visualization.__ops import get_fig_kwargs


def segmentation(
        x: np.ndarray, 
        y: np.ndarray, 
        color_onset: int = 30,
        alpha: float = 0.2,
        returns: bool = False,
        **kwargs: dict
    ) -> Tuple[Figure, np.ndarray]:

    # Check inputs
    if (x.ndim != 2) or (y.ndim not in [2,3]):
        raise ValueError("Supposed to work with 2D images (grayscale)")

    if x.shape == y.shape:
        y = np.copy(y)[None,]

    # Get figure and axis
    f,ax = get_fig_kwargs(**kwargs)

    # Obtain mask
    mask = y*np.arange(1,y.shape[0]+1)[:,None,None]
    mask = mask.max(axis=0).astype(int)

    # Obtain grid
    grid_X,grid_Y = np.meshgrid(np.arange(x.shape[1]),np.arange(x.shape[0]))
    
    # Obtain unique mask elements
    unique_elements = np.unique(y)+0.5
    colors = list(mcolors.cnames)[color_onset:color_onset+unique_elements.size]

    # Plot image in axis
    ax.imshow(x,cmap='gray')
    ax.contourf(grid_X,grid_Y,mask,unique_elements,colors=colors,alpha=alpha)

    if returns:
        return f,ax
