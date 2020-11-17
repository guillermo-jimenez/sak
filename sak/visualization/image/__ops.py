from typing import Any, List, Tuple, Union, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sak
from matplotlib.figure import Figure
import matplotlib.colors as mcolors


def segmentation(
        x: np.ndarray, 
        mask: np.ndarray, 
        ax: Optional[Union[np.ndarray, matplotlib.axes.Axes]] = None,
        returns: bool = False, 
        fig: Optional[matplotlib.figure.Figure] = None,
        **kwargs: dict
    ) -> Tuple[Figure, np.ndarray]:

    # Manage inputs
    if not isinstance(x, np.ndarray):
        x = x.cpu().detach().numpy()
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().detach().numpy()

    # Squeeze mask
    x = x.squeeze()
    mask = mask.squeeze()
    if (mask.ndim == 4) or (mask.ndim == 2):
        raise ValueError("Can only work with single-, multi-channel images in format 'CxHxW'")

    mask_single = np.zeros((mask.shape[-2],mask.shape[-1]))
    limits = []
    for c in range(mask.shape[0]):
        limits.append(c+0.5)
        mask_single += (c+1)*mask[c]
    limits.append(c+1.5)
    colors = list(mcolors.cnames)[30:30+len(limits)]

    # Plot image in axis
    ax.imshow(x,cmap='gray')
    grid_X, grid_Y = np.meshgrid(np.arange(x.shape[-1]),np.arange(x.shape[-2]))
    ax.contourf(grid_X,grid_Y,mask_single,tuple(limits),colors=tuple(colors),alpha=0.2)

    if returns: return fig,ax
