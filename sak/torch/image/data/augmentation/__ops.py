import math
import torch
import numpy
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
    def __init__(self, max_factor: int = 5):
        self.max_factor = max_factor

    def __call__(self, x: torch.Tensor):
        factor = np.random.randint(2,self.max_factor+1)
        x = torch.nn.AvgPool2d(kernel_size = factor, stride = factor, padding = factor//2)(x.clone())
        x = torch.nn.Upsample(scale_factor = factor)(x)
        return x


class EqualizeHistogram(object):
    def __init__(self, nbins: int = 5, mask = None):
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

