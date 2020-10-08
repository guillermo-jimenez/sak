import math
import torch
import numpy
import sak.data
import sak.signal
from scipy.interpolate import interp1d
import numpy as np

class RandomSpikes(object):
    def __init__(self, amplitude: float, period: int = None, randomize: bool = True):
        self.amplitude = amplitude
        self.period = period
        self.randomize = randomize

    def __call__(x: torch.Tensor):
        # Get sizes
        batch_size,channels,samples = x.shape

        # Manage inputs
        if self.period is None: 
            period = np.random.randint(60,160) # Rule of thumb
        else:
            period = int(self.period)
        if self.randomize:
            amplitude = np.random.rand(batch_size)*self.amplitude
            period = period + np.random.randint(low=-period//4, high=period//4)
        else:
            amplitude = np.array([self.amplitude]*batch_size)
        amplitude = torch.tensor(amplitude[:,None,None],dtype=x.dtype)
        onset = np.random.randint(low=0, high=int(period))
        
        # Define a randomly initialized filter bank
        noise = np.zeros((5,))
        noise[0] = np.random.uniform(-0.15,0.25)
        noise[1] = np.random.uniform(0.25,0.5)
        noise[2] = np.random.uniform(1,2)
        noise[3] = np.random.uniform(-0.5,0.25)
        noise[4] = np.random.uniform(0,0.25)

        # Interpolate to number of samples
        N = np.random.randint(5,11)
        interpolator = interp1d(np.linspace(0,1,5), noise, kind='quadratic')
        noise = interpolator(np.linspace(0,1,N))
        
        # Correct signal
        noise = sak.data.ball_scaling(sak.signal.on_off_correction(noise),metric=sak.signal.abs_max)
        noise = np.random.choice([-1,1])*noise
        
        # Generate signal-wide noise
        noise = np.concatenate((noise,np.zeros((period,))))
        noise = np.tile(noise,math.ceil(samples/noise.size))
        noise = noise[None,None,:samples]
        noise = np.tile(noise,(batch_size,channels,1))
        noise = torch.tensor(noise,dtype=x.dtype)*amplitude
        
        return x + noise