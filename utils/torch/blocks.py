import torch
import torch.nn
import utils
import utils.torch.modules
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import Conv3d
from torch.nn import ConvTranspose1d
from torch.nn import ConvTranspose2d
from torch.nn import ConvTranspose3d
from torch.nn import Unfold
from torch.nn import Fold
from torch.nn import RNNBase
from torch.nn import RNN
from torch.nn import LSTM
from torch.nn import GRU
from torch.nn import RNNCell
from torch.nn import LSTMCell
from torch.nn import GRUCell
# from torch.nn import Transformer
# from torch.nn import TransformerEncoder
# from torch.nn import TransformerDecoder
# from torch.nn import TransformerEncoderLayer
# from torch.nn import TransformerDecoderLayer
# from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Bilinear
from torch.nn import PixelShuffle
from torch.nn import Upsample
from torch.nn import UpsamplingNearest2d
from torch.nn import UpsamplingBilinear2d
# from torch.nn import Flatten
from torch.nn import MaxPool1d
from torch.nn import MaxPool2d
from torch.nn import MaxPool3d
from torch.nn import MaxUnpool1d
from torch.nn import MaxUnpool2d
from torch.nn import MaxUnpool3d
from torch.nn import AvgPool1d
from torch.nn import AvgPool2d
from torch.nn import AvgPool3d
from torch.nn import FractionalMaxPool2d
from torch.nn import LPPool1d
from torch.nn import LPPool2d
from torch.nn import AdaptiveMaxPool1d
from torch.nn import AdaptiveMaxPool2d
from torch.nn import AdaptiveMaxPool3d
from torch.nn import AdaptiveAvgPool1d
from torch.nn import AdaptiveAvgPool2d
from torch.nn import AdaptiveAvgPool3d
from utils.__ops import required
from utils.__ops import check_required

"""
Order of operations
https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
"""

class Identity(torch.nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> x = torch.randn(128, 20)
        >>> output = m(x)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Add(torch.nn.Module):
    r"""A placeholder identity operator for addition.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = Add(54, unused_argument1=0.1, unused_argument2=False)
        >>> x = torch.randn(128, 20)
        >>> y = torch.randn(128, 20)
        >>> output = m(x,y)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

class UnFlatten(torch.nn.Module):
    def __init__(self, shape: torch.Size or list or tuple):
        super(UnFlatten, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], *self.shape)


class Regularization(torch.nn.Module):
    def __init__(self, activation: dict = {}, normalization: dict = {}, dropout: dict = {}):
        super(Regularization, self).__init__()

        # Declare operations
        self.regularization = []
        self.activation = False
        self.normalization = False
        self.dropout = False

        if activation.get('name'):
            self.activation = utils.class_selector('utils.torch.activation', activation['name'])(**activation.get('arguments', {}))
        if normalization.get('name'):
            self.normalization = utils.class_selector('utils.torch.normalization', normalization['name'])(**normalization.get('arguments', {}))
        if dropout.get('name'):
            self.dropout = utils.class_selector('utils.torch.dropout', dropout['name'])(**dropout.get('arguments', {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation:
            x = self.activation(x)
        if self.normalization:
            x = self.normalization(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class GlobalAvgPooling1d(torch.nn.Module):
    def __init__(self, dim: int = None, keepdims: bool = False):
        super(GlobalAvgPooling1d, self).__init__()

        self.dim = dim
        self.keepdims = keepdims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(self.dim, self.keepdims)


class PointWiseConv1d(torch.nn.Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointWiseConv1d, self).__init__()

        # Check required inputs
        check_required(self, {'in_channels':in_channels, 'out_channels':out_channels})

        # Establish default inputs
        kwargs['groups'] = 1
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0

        # Declare operation
        self.pointwise_conv = Conv1d(in_channels, out_channels, **kwargs)

        # Initialize weights values
        initializer = utils.class_selector('torch.nn.init', kwargs.get('initializer','xavier_normal_'))
        initializer(self.pointwise_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(x)


class DepthwiseConv1d(torch.nn.Module):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConv1d, self).__init__()

        # Check required inputs
        check_required(self, {'in_channels':in_channels, 'kernel_size':kernel_size})
        
        # Establish default inputs
        kwargs['groups'] = in_channels
        kwargs['padding'] = kwargs.get('padding', (kernel_size-1)//2)
        if 'out_channels' in kwargs:
            kwargs.pop('out_channels')

        # Declare operation
        self.depthwise_conv = Conv1d(in_channels, in_channels, kernel_size, **kwargs)
        
        # Initialize weights values
        initializer = utils.class_selector('torch.nn.init', kwargs.get('initializer','xavier_normal_'))
        initializer(self.depthwise_conv.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv(x)


class SeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConv1d, self).__init__()
        
        # Check required inputs
        check_required(self, {'in_channels':in_channels,'out_channels':out_channels,'kernel_size':kernel_size})

        # Declare operations
        self.depthwise_conv = DepthwiseConv1d(in_channels, kernel_size, **kwargs)
        self.pointwise_conv = PointWiseConv1d(in_channels, out_channels, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.depthwise_conv(x)
        h = self.pointwise_conv(h)
        return h


class PointWiseConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointWiseConvTranspose1d, self).__init__()

        # Check required inputs
        check_required(self, {'in_channels':in_channels,'out_channels':out_channels})

        # Establish default inputs
        kwargs['groups'] = 1
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0

        # Declare operation
        self.pointwise_conv_transp = ConvTranspose1d(in_channels, out_channels, **kwargs)

        # Initialize weights values
        initializer = utils.class_selector('torch.nn.init', kwargs.get('initializer','xavier_normal_'))
        initializer(self.pointwise_conv_transp.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv_transp(x)


class DepthwiseConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConvTranspose1d, self).__init__()
        
        # Check required inputs
        check_required(self, {'in_channels':in_channels, 'kernel_size':kernel_size})

        # Establish default inputs
        kwargs['groups'] = in_channels
        kwargs['padding'] = kwargs.get('padding', (kernel_size-1)//2)
        if 'out_channels' in kwargs:
            kwargs.pop('out_channels')
        
        # Declare operation
        self.depthwise_conv_transp = ConvTranspose1d(in_channels, in_channels, kernel_size, **kwargs)
        
        # Initialize weights values
        initializer = utils.class_selector('torch.nn.init', kwargs.get('initializer','xavier_normal_'))
        initializer(self.depthwise_conv_transp.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv_transp(x)


class SeparableConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConvTranspose1d, self).__init__()
        
        # Declare operations
        self.depthwise_conv_transp = DepthwiseConvTranspose1d(in_channels, kernel_size, **kwargs)
        self.pointwise_conv_transp = PointWiseConvTranspose1d(in_channels, out_channels, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.depthwise_conv_transp(x)
        h = self.pointwise_conv_transp(h)
        return h


class Residual(torch.nn.Module):
    def __init__(self, in_channels: int = required, 
                       out_channels: int = required, 
                       kernel_size: int = required,
                       operation: str = required,
                       repetitions: int = required,
                       regularization: dict = {},
                      **kwargs: dict):
        super(Residual, self).__init__()
        
        # Check required inputs
        check_required(self, {'in_channels':in_channels,'out_channels':out_channels,'kernel_size':kernel_size,'operation':operation,'repetitions':repetitions})

        # Define operation to be performed
        self.repetitions = repetitions
        self.operation = utils.class_selector('utils.torch.blocks', operation)

        # Check number of repetitions is higher than 1 (otherwise why bother?)
        if repetitions < 1:
            raise ValueError("Number of repetitions must be higher than 1")

        # Stupid decoration
        __in_channels = in_channels

        # Define stack of operations
        self.operation_stack = []
        for i in range(repetitions):
            self.operation_stack.append(self.operation(in_channels, out_channels, kernel_size, **kwargs))
            if (regularization) and (i != repetitions-1):
                self.operation_stack.append(Regularization(**regularization))
            in_channels = out_channels

        # Operations
        self.operation_stack = torch.nn.Sequential(*self.operation_stack)

        # Operation if # of channels changes
        if __in_channels != out_channels:
            self.output_operation = self.operation(__in_channels, out_channels, kernel_size, **kwargs)

        # Residual
        self.addition = Add()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.clone()
        h = self.operation_stack(h)
        
        # If the number of channels of x and h does not coincide,
        # apply same transformation to x
        if x.shape[1] != h.shape[1]:
            x = self.output_operation(x)

        return self.addition(x, h) # Residual connection

