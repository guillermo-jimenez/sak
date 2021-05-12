from typing import Union, Tuple, List, Iterable, Callable, Any

import cv2
import math
import torch
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

    def __call__(self, *args: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        for i,elem_i in enumerate(args):
            assert isinstance(elem_i, torch.Tensor)

            for j,elem_j in enumerate(args):
                if i == j:
                    continue
                assert elem_i.shape[-2:] == elem_j.shape[-2:], "The shapes of the input tensors do not coincide"

        # Retrieve input shapes
        bs,ch,h,w = args[0].shape

        # Shift tensors X dimensions
        outputs = [elem.clone() for elem in args]

        for b in range(bs):
            # Obtain the number of samples to move
            shift_x = round(np.random.uniform(-self.ratio_x, self.ratio_x)*w)
            shift_y = round(np.random.uniform(-self.ratio_y, self.ratio_y)*h)

            for i,elem in enumerate(outputs):
                if shift_x >= 0: 
                    elem[b,...,:int(shift_x),:] = 0
                else:
                    elem[b,...,int(shift_x):,:] = 0

                # Shift tensors X dimensions
                if shift_y >= 0: 
                    elem[b,...,:,:int(shift_y)] = 0
                else:
                    elem[b,...,:,int(shift_y):] = 0

                # Roll tensors
                elem[b,] = torch.roll(elem[b,],(-shift_x,-shift_y),dims=(-2,-1))
            
        return tuple(outputs)


class SegmentationFlip:
    def __init__(self, proba_x: float = 0.0, proba_y: float = 0.0):
        self.proba_x = proba_x
        self.proba_y = proba_y
        assert (proba_x <= 1) and (proba_x >= 0), "Probabilities should be in the interval [0,1]"
        assert (proba_y <= 1) and (proba_y >= 0), "Probabilities should be in the interval [0,1]"

    def __call__(self, *args: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        for i,elem_i in enumerate(args):
            assert isinstance(elem_i, torch.Tensor)

            for j,elem_j in enumerate(args):
                if i == j:
                    continue
                assert elem_i.shape[-2:] == elem_j.shape[-2:], "The shapes of the input tensors do not coincide"
        
        # Retrieve input shapes
        bs,ch,h,w = args[0].shape
        
        # Obtain the number of samples to move
        flip_x = np.random.rand() <= self.proba_x
        flip_y = np.random.rand() <= self.proba_y
        
        # Shift tensors X dimensions
        outputs = []

        for elem in args:
            # Flip tensors dimensions
            if flip_x: 
                elem = torch.flip(elem,[-1])
            if flip_y: 
                elem = torch.flip(elem,[-2])
            
            # Add as output
            outputs.append(elem)
            
        return tuple(outputs)


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

    def __call__(self, *args: List[torch.Tensor]):
        for i,elem_i in enumerate(args):
            assert isinstance(elem_i, torch.Tensor)

            for j,elem_j in enumerate(args):
                if i == j:
                    continue
                assert elem_i.shape[-2:] == elem_j.shape[-2:], "The shapes of the input tensors do not coincide"
        
        # Retrieve input shapes
        bs,ch,h,w = args[0].shape
        
        # Apply transformation to each element in the batch
        scale       = (np.random.rand(self.max_iterations,bs)*self.threshold_scale+(1-self.threshold_scale/2))
        rotation    = (np.random.rand(self.max_iterations,bs)*self.threshold_rotation-self.threshold_rotation/2)
        shear       = (np.random.rand(self.max_iterations,bs)*self.threshold_shear-self.threshold_shear/2)
        translation = (np.random.rand(self.max_iterations,bs)*self.threshold_translation-self.threshold_translation/2)

        # Shift tensors X dimensions
        outputs = []

        for elem in args:
            # Output tensor
            out_elem = elem.clone()

            for it in range(np.random.randint(self.max_iterations)):
                for b in range(elem.shape[0]):
                    # Define warp matrix
                    warp_matrix = skimage.transform.AffineTransform(None,scale[it,b],rotation[it,b],shear[it,b],translation[it,b])

                    for c in range(elem.shape[1]):
                        out_elem[b,c] = torch.tensor(skimage.transform.warp(out_elem[b,c].numpy(), warp_matrix))

            # Add as output
            outputs.append(out_elem)
            
        return tuple(outputs)


class UpDownSample(object):
    def __init__(self, max_exponent: int = 2):
        self.max_exponent = max_exponent

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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

    def __call__(self, *x: Tuple[torch.Tensor]):
        out = []
        for i in range(len(x)):
            out.append(torch.tensor(skimage.exposure.rescale_intensity(x[i].numpy(),out_range=self.out_range)))
        return tuple(out)


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

class PrintCursors(object):
    def __init__(self, proba_crosshairs = 0.5, proba_horizontal_lines = 0.5, proba_vertical_lines = 0.5, 
                       proba_ruler_x = 0.05, proba_ruler_y = 0.5, scale = 5, upscaling_factor = 10, fs = 1000.):
        self.proba_crosshairs = proba_crosshairs
        self.proba_horizontal_lines = proba_horizontal_lines
        self.proba_vertical_lines = proba_vertical_lines
        self.proba_ruler_x = proba_ruler_x
        self.proba_ruler_y = proba_ruler_y
        self.scale = scale
        self.upscaling_factor = upscaling_factor
        self.fs = fs

    def __call__(self, x: torch.Tensor, y_1d: torch.Tensor = None):
        # Probabilities & other parameters
        has_crosshair    = random.uniform(0,1) > (1-self.proba_crosshairs)
        has_horizlines   = random.uniform(0,1) > (1-self.proba_horizontal_lines)
        has_vertlines    = random.uniform(0,1) > (1-self.proba_vertical_lines)
        has_ruler_x      = random.uniform(0,1) > (1-self.proba_ruler_x)
        has_ruler_y      = random.uniform(0,1) > (1-self.proba_ruler_y)
        N_horiz = np.random.choice([1,2],       p=[0.45,0.55,])*has_horizlines
        N_vert  = np.random.choice([1,2,3,4,5], p=[0.15,0.58,0.25,0.01,0.01,])*has_vertlines
        N_cross = np.random.choice([1,2,3,4],   p=[0.6,0.38,0.01,0.01])*has_crosshair
        ruler_spacing     = random.randint(10,50)
        ruler_width       = random.randint(10,20)
        ruler_onset       = random.randint(0,ruler_spacing)
        crosshair_length  = random.randint(15,25)
        crosshair_width   = np.random.choice([1,2],p=[0.2,0.8])
        vertline_width    = np.random.choice([1,2],p=[0.8,0.2])
        horizline_spacing = random.randint(5,10)

        # Avoid modifying inputs forever
        x = x.numpy().copy()
        y_1d = y_1d.numpy().copy() if (y_1d is not None) else None
        
        # Avoid issues with max values
        if x.max() <= 1:
            value = 1
        elif x.max() <= 255:
            value = 255
        else:
            raise NotImplementedError("Value not implemented yet")

        # Get dimensions
        BS,CH,H,W = x.shape

        for b in range(BS):
            for c in range(CH):
                # To be generated
                if y_1d is not None:
                    crossings = sak.signal.find_peaks(y_1d[b,c],fs=self.fs,scale=self.scale,upscale=self.upscaling_factor)
                    locations = list(zip(crossings,y_1d[b,c,crossings].astype(int)))
                else:
                    locations = [(random.randint(0,W),random.randint(0,H)) for _ in range(15)]
                if len(locations) == 0:
                    locations = [(random.randint(0,W),random.randint(0,H)) for _ in range(15)]

                unique_x = np.unique([loc[0] for loc in locations]).tolist()
                unique_y = np.unique([loc[1] for loc in locations]).tolist()

                for _ in range(N_cross):
                    loc_x,loc_y = random.choice(locations)
                    loc_x = int(loc_x + random.normalvariate(0,1)*10)
                    loc_y = int(loc_y + random.normalvariate(0,1)*10)

                    n = random.randint(1,5)
                    x_space = x[b,c,loc_y-crosshair_width-n:loc_y+crosshair_width+n,
                                loc_x-crosshair_width-n:loc_x+crosshair_width+n].copy()
                    x[b,c,loc_y-crosshair_width:loc_y+crosshair_width,
                      loc_x-crosshair_length:loc_x+crosshair_length,] = value
                    x[b,c,loc_y-crosshair_length:loc_y+crosshair_length,
                      loc_x-crosshair_width:loc_x+crosshair_width,] = value

                    if random.uniform(0,1) > (1-0.5):
                        x[b,c,loc_y-crosshair_width-n:loc_y+crosshair_width+n,
                          loc_x-crosshair_width-n:loc_x+crosshair_width+n] = x_space

                for _ in range(N_vert):
                    loc_x = random.choice(unique_x) + random.randint(-5,5)
                    x[b,c,:,loc_x-vertline_width:loc_x+vertline_width] = value

                for _ in range(N_horiz):
                    loc_y = random.choice(unique_y) + random.randint(-5,5)
                    dashes = sak.signal.pulse_train(W,horizline_spacing,random.randint(0,horizline_spacing)).astype(bool)
                    x[b,c,loc_y-1:loc_y+1,dashes] = value

                if has_ruler_x:
                    for i,loc in enumerate(range(ruler_onset,W,ruler_spacing)):
                        x[b,c,-int(ruler_width*(1.5**(i%random.choice([5,10,20])==0))):,loc-1:loc+1] = value
                if has_ruler_x or has_ruler_y:
                    for i,loc in enumerate(range(ruler_onset,H,ruler_spacing)):
                        x[b,c,loc-1:loc+1,-int(ruler_width*(1.5**(i%random.choice([5,10,20])==0))):] = value
        
        return torch.tensor(x)
