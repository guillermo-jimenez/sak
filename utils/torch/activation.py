import torch 
import torch.nn
from torch.nn import ELU
from torch.nn import Hardshrink
from torch.nn import Hardtanh
from torch.nn import LeakyReLU
from torch.nn import LogSigmoid
from torch.nn import MultiheadAttention
from torch.nn import PReLU
from torch.nn import ReLU
from torch.nn import ReLU6
from torch.nn import RReLU
from torch.nn import SELU
from torch.nn import CELU
from torch.nn import GELU
from torch.nn import Sigmoid
from torch.nn import Softplus
from torch.nn import Softshrink
from torch.nn import Softsign
from torch.nn import Tanh
from torch.nn import Tanhshrink
from torch.nn import Threshold
from torch.nn import Softmin
from torch.nn import Softmax
from torch.nn import Softmax2d
from torch.nn import LogSoftmax
from torch.nn import AdaptiveLogSoftmaxWithLoss

class none(torch.nn.Module):
    r"""Does not apply any non-linear activation"""

    def __init__(self,*args,**kwargs):
        super(none, self).__init__()
        pass

    def forward(self, x):
        return x

class linear(torch.nn.Module):
    r"""Does not apply any non-linear activation
    """

    def __init__(self,):
        super(linear, self).__init__()
        pass

    def forward(self, x):
        return x

