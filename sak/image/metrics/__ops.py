from typing import Tuple
import numpy as np
import cv2
import skimage
import skimage.exposure
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull


"""Borrowed from https://github.com/scikit-image/scikit-image/blob/2db874cc6485186b08d296887a8a91bb2f8865ec/skimage/metrics/set_metrics.py#L4-L51
   as it has not yet been implemented in my scikit-image version (0.17.2 vs 0.18.0)"""
def hausdorff_distance(image0: np.ndarray, image1: np.ndarray) -> float:
    """Calculate the Hausdorff distance between nonzero elements of given images.
    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.
    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of nonzero pixels in
        ``image0`` and ``image1``, using the Euclidian distance.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=np.bool)
    >>> image_b = np.zeros(shape, dtype=np.bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_distance(image_a, image_b)
    3.0
    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))


# https://arxiv.org/pdf/1908.02994.pdf
def convexity(mask: np.ndarray) -> float:
    # Convex hull
    hull = ConvexHull(np.array(np.where(mask)).T)
    hull_area = hull.volume
    
    # Mask
    contours,_ = cv2.findContours(mask.astype('uint8'), 1, cv2.CHAIN_APPROX_SIMPLE)
    mask_area = sum([cv2.contourArea(c) for c in contours])
    
    return mask_area/hull_area

# https://arxiv.org/pdf/1908.02994.pdf
def simplicity(mask: np.ndarray) -> float:
    # Mask
    contours,_ = cv2.findContours(mask.astype('uint8'), 1, cv2.CHAIN_APPROX_NONE)
    mask_area = sum([cv2.contourArea(c) for c in contours])
    mask_perimeter = sum([c.shape[0] for c in contours])
    
    return ((4*np.pi*mask_area)**(1/2))/mask_perimeter

def elements(mask: np.ndarray) -> int:
    contours,_ = cv2.findContours(mask.astype('uint8'), 1, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def instance_metrics(input_elements: float, target_elements: float) -> Tuple[float,float,float]:
    truepositive  = np.abs(target_elements-max([(target_elements-input_elements),0]))
    falsepositive = max([input_elements-target_elements,0])
    falsenegative = max([target_elements-input_elements,0])
    return truepositive, falsepositive, falsenegative

def precision(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+fp)

def recall(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+fn)

def f1_score(tp: int, fp: int, fn: int) -> float:
    return tp/(tp+(fp+fn)/2)

def get_boundaries(mask: np.ndarray, mode: int = cv2.RETR_TREE, method: int = cv2.CHAIN_APPROX_TC89_KCOS) -> np.ndarray:
    if mask.ndim == 2:
        mask = mask[None,]
        
    mask = skimage.exposure.rescale_intensity(mask, out_range='uint8')
    
    output_contours = []
    for i,channel in enumerate(mask):
        # Find contour
        cnt = cv2.findContours(channel,mode,method)
        
        # Get largest connected component
        cnt = cnt[0][0]
        
        # Squeeze irrelevant dimensions
        cnt = cnt.squeeze()
        
        # Close circle
        cnt = np.vstack((cnt,cnt[0,:]))
        
        # Return
        output_contours.append(cnt)
        
    return output_contours
    
    