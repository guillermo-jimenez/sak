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
    def __init__(self, matrix: np.ndarray = None, scale: float = None, rotation: float = None, shear: float = None, translation: float = None):
        self.matrix = matrix
        self.scale = scale
        self.rotation = rotation
        self.shear = shear
        self.translation = translation

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        # Output tensor
        out_x = torch.empty_like(x)
        out_y = torch.empty_like(y)

        # Define warp matrix
        matrix,scale,rotation,shear,translation = (None,None,None,None,None)
        if      self.matrix is not None: matrix      =      self.matrix*(np.random.rand()*0.5 + 0.75)*np.random.choice([-1,1])
        if       self.scale is not None: scale       =       self.scale*(np.random.rand()*0.5 + 0.75)*np.random.choice([-1,1])
        if    self.rotation is not None: rotation    =    self.rotation*(np.random.rand()*0.5 + 0.75)*np.random.choice([-1,1])
        if       self.shear is not None: shear       =       self.shear*(np.random.rand()*0.5 + 0.75)*np.random.choice([-1,1])
        if self.translation is not None: translation = self.translation*(np.random.rand()*0.5 + 0.75)*np.random.choice([-1,1])
        warp_matrix = skimage.transform.AffineTransform(matrix,scale,rotation,shear,translation)

        # Apply transformation to each element in the batch
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                out_x[b,c] = torch.tensor(skimage.transform.warp(x[b,c].numpy(), warp_matrix))
            for c in range(y.shape[1]):
                out_y[b,c] = torch.tensor(skimage.transform.warp(y[b,c].numpy(), warp_matrix))

        return out_x, out_y


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

