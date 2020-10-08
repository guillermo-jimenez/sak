import math
import torch
import numpy
import sak.data
import sak.signal
from scipy.interpolate import interp1d
import numpy as np

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
            period = np.random.randint(100,250,size=batch_size) # Rule of thumb
        else: 
            period = np.array([period]*batch_size,astype=int)
        amplitude = np.random.rand(batch_size)*self.amplitude
        amplitude = (sample_max*torch.tensor(amplitude,dtype=x.dtype))[:,None,None]
        period = period + np.random.randint(low=-period//4, high=period//4,size=batch_size)
        
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