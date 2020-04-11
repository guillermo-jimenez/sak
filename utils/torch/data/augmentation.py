import torch
import numpy as np
from torchvision.transforms import *

# Check required arguments as keywords
from utils.__ops import required
from utils.__ops import check_required


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

    def __init__(self, samples: int = required, dims: list = [-1]):
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
            sample (torch.Tensor): tensor to apply transformation on. Format batch_size x channels x samples
        """
        Noise         = torch.zeros_like(X)

        SignalPower   = torch.mean((X - torch.mean(X, dim=-1, keepdim=True))**2, dim=-1, keepdim=True)
        SNRdb         = self.SNR + (self.SNR/10)*np.random.uniform(-1,1)
        Freq          = self.powerline_freq # Europe: 50Hz, USA: 60Hz

        NormFreq      = 2.*np.pi*Freq/self.signal_freq # Normalized frequency
        NoisePower    = SignalPower/10**(SNRdb/10.)
        Amplitude     = np.sqrt(2*NoisePower)

        for b in range(X.shape[0]):
            for c in range(X.shape[1]):
                Noise[b,c,] = Amplitude[b,c]*torch.sin(NormFreq*torch.arange(X.shape[-1]).type(X.type()) + np.pi*np.random.uniform(-1,1))
        
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

        SignalPower   = torch.mean((X - torch.mean(X, dim=-1, keepdim=True))**2, dim=-1, keepdim=True)
        SNRdb         = self.SNR + (self.SNR/10)*np.random.uniform(-1,1)
        Freq          = self.baseline_freq + (self.baseline_freq/4)*np.random.uniform(-1,1)# Europe: 50Hz, USA: 60Hz

        NormFreq      = 2.*np.pi*Freq/self.signal_freq # Normalized frequency
        NoisePower    = SignalPower/10**(SNRdb/10.)
        Amplitude     = np.sqrt(2*NoisePower)
        
        for b in range(X.shape[0]):
            for c in range(X.shape[1]):
                Noise[b,c,] = Amplitude[b,c]*torch.sin(NormFreq*torch.arange(X.shape[-1]).type(X.type()) + np.pi*np.random.uniform(-1,1))

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
        Noise         = (self.percentage*X.abs().max())*torch.randn(*X.shape[0:-1])[:,None].type(X.type())

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

        
