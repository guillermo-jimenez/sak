import torch
import torch.nn
from torch.nn import Dropout
from torch.nn import Dropout2d
from torch.nn import Dropout3d
from torch.nn import AlphaDropout

class none(torch.nn.Module):
    """Does not apply dropout"""

    def __init__(self,*args,**kwargs):
        super(none, self).__init__()
        pass

    def forward(self, x):
        return x

class Dropout1d(torch.nn.Module):
    """Applies one-dimensional spatial dropout"""
    def __init__(self, p: float):
        super(Dropout1d, self).__init__()
        assert (p >= 0) and (p <= 1)
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add a dimension for 2D to work -> format BxCxHxW
        x = x.unsqueeze(-1) 
        x = torch.nn.Dropout2d(self.p)(x).squeeze(-1)
        return x

