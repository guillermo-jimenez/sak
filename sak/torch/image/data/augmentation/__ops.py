from typing import Union, Tuple, List, Iterable, Callable, Any

import cv2
import math
import torch
import numpy
import random
import sak.data
import sak.signal
import skimage.transform
import skimage.exposure
import numpy as np

class HistogramMatching(object):
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor):
        # Output tensor
        out = torch.empty_like(x)

        # Elements in the batch
        elements = np.arange(x.shape[0])

        # Define source and template randomly
        histogram_pairs = (np.random.permutation(elements),np.random.permutation(elements))

        # Randomly assign all matched histograms in the batch
        for (sample_from, sample_to) in zip(*histogram_pairs):
            x_from = x[sample_from].permute([1,2,0]).numpy()
            x_to   = x[sample_to].permute([1,2,0]).numpy()
            out[sample_from] = torch.tensor(skimage.exposure.match_histograms(x_from,x_to)).permute([2,0,1])

        return out


class SegmentationShift:
    def __init__(self, ratio_x: float = 0.0, ratio_y: float = 0.0):
        self.ratio_x = ratio_x
        self.ratio_y = ratio_y
        assert (ratio_x <= 1) and (ratio_x >= 0), "Ratios should be in the interval [0,1]"
        assert (ratio_y <= 1) and (ratio_y >= 0), "Ratios should be in the interval [0,1]"

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-2:] == y.shape[-2:], "The shapes of the input tensors do not coincide"
        
        # Retrieve input shapes
        bs,ch,h,w = x.shape
        
        # Obtain the number of samples to move
        shift_x = round(np.random.uniform(-self.ratio_x, self.ratio_x)*w)
        shift_y = round(np.random.uniform(-self.ratio_y, self.ratio_y)*h)
        
        # Shift tensors X dimensions
        if shift_x > 0: 
            x[...,:int(shift_x),:] = 0
            y[...,:int(shift_x),:] = 0
        else:
            x[...,int(shift_x):,:] = 0
            y[...,int(shift_x):,:] = 0

        # Shift tensors X dimensions
        if shift_y > 0: 
            x[...,:,:int(shift_y)] = 0
            y[...,:,:int(shift_y)] = 0
        else:
            x[...,:,int(shift_y):] = 0
            y[...,:,int(shift_y):] = 0
            
        # Roll tensors
        x = torch.roll(x,(-shift_x,-shift_y),dims=(-2,-1))
        y = torch.roll(y,(-shift_x,-shift_y),dims=(-2,-1))
        
        return x,y


class SegmentationFlip:
    def __init__(self, proba_x: float = 0.0, proba_y: float = 0.0):
        self.proba_x = proba_x
        self.proba_y = proba_y
        assert (proba_x <= 1) and (proba_x >= 0), "Probabilities should be in the interval [0,1]"
        assert (proba_y <= 1) and (proba_y >= 0), "Probabilities should be in the interval [0,1]"

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        assert x.shape[-2:] == y.shape[-2:], "The shapes of the input tensors do not coincide"
        
        # Retrieve input shapes
        bs,ch,h,w = x.shape
        
        # Obtain the number of samples to move
        flip_x = np.random.rand() <= self.proba_x
        flip_y = np.random.rand() <= self.proba_y
        
        # Flip tensors dimensions
        if flip_x: 
            x = torch.flip(x,[-1])
            y = torch.flip(y,[-1])
        if flip_y: 
            x = torch.flip(x,[-2])
            y = torch.flip(y,[-2])
        
        return x,y


class ClipIntensities:
    def __init__(self, threshold: float = 0.0, mode: str = 'max'):
        assert (threshold <= 1) and (threshold >= 0), "Threshold should be in the interval [0,1]"
        assert mode.lower() in ['min', 'max'], "Mode can only be in ['min', 'max']"
        self.threshold = threshold
        self.ismax = 1 if mode.lower() == 'max' else 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Get maximum value (very dirty)
        maxval = 255 if x.max() > 1 else 1.
        
        # A bit more flexibility
        threshold = np.random.uniform(-self.threshold/10,self.threshold/10)+self.threshold
        
        if self.ismax:
            return x.clamp_max(maxval*threshold)
        else:
            return x.clamp_min(maxval*threshold)


class BlurImage:
    def __init__(self, kernel_size: Union[int, Iterable] = None, background_value: float = None):
        self.kernel_size = kernel_size
        self.background_value = background_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Get background mask
        if self.background_value is not None:
            background_mask = (x == (self.background_value))

        # Output structure
        out_x = torch.zeros_like(x)
        
        # Iterate over samples and channels
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                if self.kernel_size is None:
                    kernel_size = [random.randrange(3,7+1,2) for _ in range(x.ndim-2)]
                else:
                    if isinstance(self.kernel_size, int):
                        kernel_size = [self.kernel_size for _ in range(x.ndim-2)]
                    elif isinstance(self.kernel_size, np.ndarray):
                        assert self.kernel_size.size == x.ndim-2
                        kernel_size = self.kernel_size.tolist()
                    else:
                        assert len(self.kernel_size) == x.ndim-2
                        kernel_size = list(self.kernel_size)
                
                out_x[b,c,] = torch.tensor(cv2.blur(x[b,c].numpy(),tuple(kernel_size)))
                
        if self.background_value is not None:
            out_x[background_mask] = self.background_value
        
        return out_x
        
        
class EnhanceBorders:
    def __init__(self, background_value: float = None, sigma_s: float = None, sigma_r: float = None):
        self.background_value = background_value
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Get background mask
        if self.background_value is not None:
            background_mask = (x == (self.background_value))
            
        # Output structure
        out_x = torch.zeros_like(x)
        
        # Iterate over samples and channels
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                out_x[b,c,] = torch.tensor(
                    cv2.detailEnhance(
                        np.repeat(x[b,c].numpy()[...,None],3,-1),
                        None,
                        self.sigma_s,
                        self.sigma_r,
                    )
                )[:,:,0]
                
        if self.background_value is not None:
            out_x[background_mask] = self.background_value
        
        return out_x
        

class AdjustGamma(object):
    def __init__(self, gamma: float = 2, noise: float = 0.5):
        self.gamma = gamma
        self.noise = noise

    def __call__(self, x: torch.Tensor):
        # Output tensor
        out = torch.empty_like(x)

        # Apply transformation to each element in the batch
        for i in range(x.shape[0]):
            x_i = x[i].permute([1,2,0]).numpy()
            g = max([self.gamma+np.random.randn()*self.noise,0])
            out[i] = torch.tensor(skimage.exposure.adjust_gamma(x_i, gamma=g)).permute([2,0,1])

        return out


class AffineTransform(object):
    def __init__(self, max_iterations: int = 3, threshold_scale: float = 0.1, threshold_rotation: float = 0.1, threshold_shear: float = 0.1, threshold_translation: float = 0.1):
        self.max_iterations = max_iterations
        self.threshold_scale = threshold_scale
        self.threshold_rotation = threshold_rotation
        self.threshold_shear = threshold_shear
        self.threshold_translation = threshold_translation

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        # Output tensor
        out_x = x.clone()
        out_y = y.clone()

        # Apply transformation to each element in the batch
        scale       = (np.random.rand(self.max_iterations,x.shape[0])*self.threshold_scale+(1-self.threshold_scale/2))
        rotation    = (np.random.rand(self.max_iterations,x.shape[0])*self.threshold_rotation-self.threshold_rotation/2)
        shear       = (np.random.rand(self.max_iterations,x.shape[0])*self.threshold_shear-self.threshold_shear/2)
        translation = (np.random.rand(self.max_iterations,x.shape[0])*self.threshold_translation-self.threshold_translation/2)
        for it in range(np.random.randint(self.max_iterations)):
            for b in range(x.shape[0]):
                # Define warp matrix
                warp_matrix = skimage.transform.AffineTransform(None,scale[it,b],rotation[it,b],shear[it,b],translation[it,b])

                for c in range(x.shape[1]):
                    out_x[b,c] = torch.tensor(skimage.transform.warp(out_x[b,c].numpy(), warp_matrix))
                for c in range(y.shape[1]):
                    out_y[b,c] = torch.tensor(skimage.transform.warp(out_y[b,c].numpy(), warp_matrix))

        return out_x, out_y


class UpDownSample(object):
    def __init__(self, max_exponent: int = 2):
        self.max_exponent = max_exponent

    def __call__(self, x: torch.Tensor):
        factor = random.randint(1,self.max_exponent)
        x = torch.nn.AvgPool2d(kernel_size = 2**factor, stride = 2**factor, padding = max([factor//2-1,0]))(x.clone())
        x = torch.nn.Upsample(scale_factor = 2**factor)(x)
        return x


class EqualizeHistogram(object):
    def __init__(self, nbins: int = 256, mask = None):
        self.nbins = nbins
        self.mask = mask

    def __call__(self, x: torch.Tensor):
        # Output structure
        out_x = torch.empty_like(x)

        # Retrieve max scale
        max_scale = x.max()
        if   x.max() <= 1:
            max_scale = 1
        elif x.max() <= 255:
            max_scale = 255

        # Perform operation
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                out = skimage.exposure.equalize_hist(x[b,c].numpy())
                out = skimage.exposure.rescale_intensity(out,out_range='float32')
                out_x[b,c] = torch.tensor(out*max_scale)

        return out_x


class SaltAndPepperNoise(object):
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def __call__(self, x: torch.Tensor):
        # Output structure
        out_x = torch.empty_like(x)
        maximum = x.max()
        max_val = maximum*(1-self.threshold)
        min_val = maximum*(1-self.threshold)
        
        # Generate mask
        noise = torch.rand_like(x)
    
        # Retrieve max scale
        max_scale = x.max()
        if   x.max() <= 1:
            max_scale = 1
        elif x.max() <= 255:
            max_scale = 255

        # Perform operation
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                out = skimage.exposure.equalize_hist(x[b,c].numpy())
                out = skimage.exposure.rescale_intensity(out,out_range='float32')
                out_x[b,c] = torch.tensor(out*max_scale)

        return out_x


class RescaleIntensity(object):
    def __init__(self, out_range: str = 'float32'):
        self.out_range = out_range

    def __call__(self, x: torch.Tensor):
        return torch.tensor(skimage.exposure.rescale_intensity(x.numpy(),out_range=self.out_range))


class SkimageLambda(object):
    def __init__(self, cls: str, arguments: dict = {}):
        self.lmbda = sak.class_selector(cls)
        self.arguments = arguments

    def __call__(self, x: torch.Tensor):
        # Output tensor
        out = torch.empty_like(x)

        # Apply transformation to each element in the batch
        for i in range(out.shape[0]):
            x_i = x[i].permute([1,2,0]).numpy()
            out[i] = torch.tensor(self.lmbda(x_i, **self.arguments)).permute([2,0,1])

        return out

