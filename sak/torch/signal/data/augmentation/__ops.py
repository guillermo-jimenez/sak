from typing import Dict, Tuple, List
import math
import torch
import numpy
import sak.data
import sak.signal
from scipy.interpolate import interp1d
import numpy as np

# Check required arguments as keywords
from sak.__ops import class_selector
from sak.__ops import required
from sak.__ops import check_required

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

    
class RandomSpikes(object):
    def __init__(self, amplitude: float, period: int = None):
        self.amplitude = amplitude
        self.period = period

    def __call__(self, x: torch.Tensor):
        # Get sizes
        batch_size,channels,samples = x.shape

        # Get the absolute maxima of each sample
        sample_max = torch.abs(x).max(-1).values.max(-1).values

        # Manage inputs
        if self.period is None: 
            period = np.random.randint(100,250,size=batch_size,dtype=int) # Rule of thumb
        else: 
            period = np.array([period]*batch_size,astype=int)
        amplitude = np.random.rand(batch_size)*self.amplitude
        amplitude = (sample_max*torch.tensor(amplitude,dtype=x.dtype))[:,None,None]
        period = period + np.random.randint(low=-np.median(period)//4, high=np.median(period)//4,size=batch_size,dtype=int)
        
        # Define a randomly initialized filter bank
        spikes = np.zeros((batch_size,channels,5,))
        spikes[:,0,0] = np.random.uniform(-0.15,0.25,size=batch_size)
        spikes[:,0,1] = np.random.uniform(0.25,0.5,size=batch_size)
        spikes[:,0,2] = np.random.uniform(1,2,size=batch_size)
        spikes[:,0,3] = np.random.uniform(-0.5,0.25,size=batch_size)
        spikes[:,0,4] = np.random.uniform(0,0.25,size=batch_size)

        # Interpolate to number of samples
        N = np.random.randint(5,11)
        interpolator = interp1d(np.linspace(0,1,5), spikes, kind='quadratic')
        spikes = interpolator(np.linspace(0,1,N))

        # Correct signal
        for i,sample in enumerate(spikes):
            for j,functional in enumerate(sample):
                # On/off correction
                spikes[i,j,:] += np.linspace(-functional[0],-functional[-1],functional.size)
                # Ball scaling
        spikes = spikes/(np.max(np.abs(spikes),-1,keepdims=True)+np.finfo(spikes.dtype).eps)
        spikes = spikes*np.random.choice([-1,1],size=batch_size)[:,None,None]

        # Generate signal-wide noise
        noise = torch.zeros_like(x,dtype=x.dtype)
        for i,spike in enumerate(spikes):
            sample_noise = np.concatenate((spike,np.zeros((channels,period[i]))),axis=-1)
            sample_noise = np.tile(sample_noise,math.ceil(samples/sample_noise.shape[-1]))
            noise[i] = torch.tensor(sample_noise[:,:samples],dtype=x.dtype)
        noise *= amplitude

        return x + noise


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

        
