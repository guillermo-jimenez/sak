from typing import Dict, Tuple, List
import torch
import torch.nn
import random
import numpy as np
import torchvision.transforms

from collections.abc import Sequence

# Check required arguments as keywords
from sak.__ops import class_selector
from sak.__ops import required
from sak.__ops import check_required


class none(object):
    """Do not add any noise to the signal"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X


class AugmentationComposer(object):
    """Compose a random transform according to pre-specified config file"""

    def __init__(self, **dic):
        self.operations = self.__get_operation(dic)

    def __call__(self, *args: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        return self.operations(*args)

    def __get_operation(self, operation):
        if "transforms" in operation:
            ops = []
            for transform in operation["transforms"]:
                ops.append(self.__get_operation(transform))
            return class_selector(operation["class"])(ops)
        else:
            if isinstance(operation["arguments"], list):
                return class_selector(operation["class"])(*operation["arguments"])
            elif isinstance(operation["arguments"], dict):
                return class_selector(operation["class"])(**operation["arguments"])


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomTransforms:
    """Base class for a list of transformations with randomness

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (list or tuple or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, *img):
        if self.p < torch.rand(1):
            return img
        for t in self.transforms:
            img = t(*img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list. This transform does not support torchscript.
    """
    def __call__(self, *args):
        t = random.choice(self.transforms)
        return t(*args)


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order. This transform does not support torchscript.
    """
    def __call__(self, *img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](*img)
        return img


class AdditiveWhiteGaussianNoise(object):
    """Add additive white gaussian noise to a signal

    Args:
        SNR (float): Signal-to-noise ratio of the signal
    """

    def __init__(self, SNR: float = required, p : [0, 1] = 0.1):
        self.SNR = SNR
        self.p = p
        check_required(self, self.__dict__)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample (torch.Tensor): tensor to apply transformation on. Format batch_size x channels x samples
        """
        SignalPower   = torch.mean((X - torch.mean(X, dim=-1, keepdim=True))**2, dim=-1, keepdim=True).type(X.type())
        SNRdb         = self.SNR + (self.SNR*self.p)*np.random.uniform(-1,1) # Add some uncertainty in the SNR
        NoisePower    = SignalPower/10**(SNRdb/10.)
        Noise         = np.sqrt(NoisePower)*torch.randn_like(X)

        return X + Noise


class SegmentationErrors(object):
    """Add additive white gaussian noise to a signal

    Args:
        SNR (float): Signal-to-noise ratio of the signal
        dims (list): axes over which to roll
    """

    def __init__(self, samples: int = required, dims: List = [-1]):
        self.samples = samples
        self.dims = dims
        check_required(self, self.__dict__)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample (torch.Tensor): tensor to apply transformation on. Format batch_size x channels x samples
        """
        return torch.roll(X, np.random.randint(-self.samples, self.samples), dims=self.dims)


class PowerlineNoise(object):
    """Add powerline noise noise to a signal

    Args:
        SNR (float): Signal-to-noise ratio of the signal
        signal_freq (float): sampling frequency of the signal
        powerline_freq (50. or 60.): frequency of the powerline
    """

    def __init__(self, SNR: float = required, signal_freq: float = required, powerline_freq: (50., 60.) = required):
        self.SNR = SNR
        self.powerline_freq = powerline_freq
        self.signal_freq = signal_freq
        check_required(self, self.__dict__)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample (torch.Tensor): tensor to apply transformation on.
        """
        Noise         = torch.ones_like(X)
        Noise[...,:]  = torch.arange(X.shape[-1])

        SignalPower   = torch.mean((X - torch.mean(X, dim=-1, keepdim=True))**2, dim=-1, keepdim=True)
        SNRdb         = self.SNR + (self.SNR/10)*np.random.uniform(-1,1)
        Freq          = self.powerline_freq # Europe: 50Hz, USA: 60Hz

        NormFreq      = 2.*np.pi*Freq/self.signal_freq # Normalized frequency
        NoisePower    = SignalPower/10**(SNRdb/10.)
        Amplitude     = np.sqrt(2*NoisePower)

        Noise         = (Amplitude*torch.sin(NormFreq*Noise + np.pi*np.random.uniform(-1,1))).type(X.type())
        
        return X + Noise


class BaselineNoise(object):
    """Add baseline noise to a signal

    Args:
        SNR (float): Signal-to-noise ratio of the signal
        signal_freq (float): sampling frequency of the signal
        baseline_freq (float): frequency of the baseline
    """

    def __init__(self, SNR: float = required, signal_freq: float = required, baseline_freq: float = required):
        self.SNR = SNR
        self.baseline_freq = baseline_freq
        self.signal_freq = signal_freq
        check_required(self, self.__dict__)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        Noise         = torch.zeros_like(X)
        Noise[...,:]  = torch.arange(X.shape[-1])

        SignalPower   = torch.mean((X - torch.mean(X, dim=-1, keepdim=True))**2, dim=-1, keepdim=True)
        SNRdb         = self.SNR + (self.SNR/10)*np.random.uniform(-1,1)
        Freq          = self.baseline_freq + (self.baseline_freq/4)*np.random.uniform(-1,1)# Europe: 50Hz, USA: 60Hz

        NormFreq      = 2.*np.pi*Freq/self.signal_freq # Normalized frequency
        NoisePower    = SignalPower/10**(SNRdb/10.)
        Amplitude     = np.sqrt(2*NoisePower)

        Noise         = (Amplitude*torch.sin(NormFreq*Noise + np.pi*np.random.uniform(-1,1))).type(X.type())

        return X + Noise


class AmplifierSaturation(object):
    """Add amplifier saturation

    Args:
        percentile (percentage): percentile of the amplifier saturation
    """

    def __init__(self, percentile: [0., 1.] = required):
        self.percentile = percentile
        check_required(self, self.__dict__)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        Noise         = torch.zeros_like(X)
        Percentile    = self.percentile + (self.percentile/4)*np.random.uniform(-1, 1) # Random percentile change

        if torch.max(X - torch.median(X)) != torch.max(torch.abs(X - torch.mean(X))):
            Noise[X <= np.percentile(X,Percentile)]     = np.percentile(X,Percentile) - X[X <= np.percentile(X,Percentile)]
        else:
            Noise[X >= np.percentile(X,100-Percentile)] = np.percentile(X,100-Percentile) - X[X >= np.percentile(X,100-Percentile)]

        return X + Noise

    
class ChangeAmplitude(object):
    """Change amplitude (baseline voltage, y axis)

    Args:
        percentage (percentage): percentage of the amplifier saturation
    """

    def __init__(self, percentage: [0., 1.] = 1):
        self.percentage = percentage
        check_required(self, self.__dict__)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        Noise         = (self.percentage*X.abs().max())*torch.randn(*X.shape[0:-1])[...,None].type(X.type())

        return X + Noise

    
# class RandomRegularSpikes(object):
#     """Add additive white gaussian noise to a signal
#     """

#     def __init__(self, SNR: float = required, T: float = required):
#         """
#         Args:
#             SNR (float): Signal-to-noise ratio of the signal
#             T (float): Period of the spikes
#         """
#         self.SNR = SNR
#         self.T = T
#         check_required(self, self.__dict__)

#     def __call__(self, X: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             sample (torch.Tensor): tensor to apply transformation on. Format batch_size x channels x samples
#         """
#         SignalPower   = torch.mean((X - torch.mean(X, dim=-1, keepdim=True))**2, dim=-1, keepdim=True)
#         SNRdb         = self.SNR + (self.SNR/10)*np.random.uniform(-1, 1)
#         T             = self.T + (self.T/4)*np.random.randint(-1,1)

#         NoisePower    = SignalPower/10**(SNRdb/10.)
#         Amplitude     = np.sqrt(NoisePower*T)
#         Noise         = torch.zeros_like(X)
#         Noise[::T]    = Amplitude
        
#         return X + Noise

        
