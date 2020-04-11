import torch
import torch.nn
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm3d
from torch.nn import GroupNorm
from torch.nn import SyncBatchNorm
from torch.nn import InstanceNorm1d
from torch.nn import InstanceNorm2d
from torch.nn import InstanceNorm3d
from torch.nn import LayerNorm
from torch.nn import LocalResponseNorm

class none(torch.nn.Module):
    r"""Does not apply any normalization"""

    def __init__(self,*args,**kwargs):
        super(none, self).__init__()
        pass

    def forward(self, x):
        return x

