from torch import Tensor
from torch.nn import Module
from torch.nn import Identity
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm3d
from torch.nn import Dropout2d
from torch.nn import Dropout3d
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import Conv3d
from torch.nn import ConvTranspose1d
from torch.nn import ConvTranspose2d
from torch.nn import ConvTranspose3d
from torch.nn import AdaptiveAvgPool1d
from torch.nn import AdaptiveAvgPool2d
from torch.nn import AdaptiveAvgPool3d
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU

from torch.nn.functional import interpolate
from .composers import Sequential
from .composers import Parallel
from sak import class_selector
from sak.__ops import required
from sak.__ops import check_required

class ImagePooling1d(Sequential):
    def __init__(self, in_channels: int = required, out_channels: int = required):
        super(ImagePooling1d, self).__init__()
        self.pooling = AdaptiveAvgPool1d(1)
        self.convolution = SeparableConv1d(in_channels, out_channels, 1, bias=False)
        self.batchnorm = BatchNorm1d(out_channels)
        self.relu = ReLU(inplace=True)

        # Check required inputs
        check_required(self, {"in_channels":in_channels, "out_channels":out_channels})

    def forward(self, x):
        size = x.shape[2:]
        x = self.pooling(x)
        x = self.convolution(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = interpolate(x.unsqueeze(-1), size=(*size,1), mode='bilinear', align_corners=False).squeeze(-1)
        return x


class PointWiseConv1d(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointWiseConv1d, self).__init__()

        # Check required inputs
        check_required(self, {"in_channels":in_channels, "out_channels":out_channels})

        # Establish default inputs
        kwargs["groups"] = 1
        kwargs["kernel_size"] = 1
        kwargs["padding"] = 0

        # Declare operation
        self.pointwise_conv = Conv1d(in_channels, out_channels, **kwargs)

        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.pointwise_conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(x)


class AttentionRefinementModule(Module):
    def __init__(self, channels: int = required, **kwargs: dict):
        super(AttentionRefinementModule, self).__init__()

        # Check required inputs
        check_required(self, {"channels":channels})

        # Establish default inputs
        self.operations = [
            AdaptiveAvgPool1d(1),
            PointWiseConv1d(channels,channels),

        ]
        self.operations = Sequential(*self.operations)

        # Declare operation
        self.pointwise_conv = Parallel((Identity(),))

        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.pointwise_conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(x)


class DepthwiseConv1d(Module):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConv1d, self).__init__()

        # Check required inputs
        check_required(self, {"in_channels":in_channels, "kernel_size":kernel_size})
        
        # Establish default inputs
        kwargs["groups"] = in_channels
        kwargs["padding"] = kwargs.get("padding", (kernel_size-1)//2)
        if "out_channels" in kwargs:
            kwargs.pop("out_channels")

        # Declare operation
        self.depthwise_conv = Conv1d(in_channels, in_channels, kernel_size, **kwargs)
        
        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.depthwise_conv.weight)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.depthwise_conv(x)


class SeparableConv1d(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConv1d, self).__init__()
        
        # Check required inputs
        check_required(self, {"in_channels":in_channels,"out_channels":out_channels,"kernel_size":kernel_size})

        # Declare operations
        self.depthwise_conv = DepthwiseConv1d(in_channels, kernel_size, **kwargs)
        self.pointwise_conv = PointWiseConv1d(in_channels, out_channels, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        h = self.depthwise_conv(x)
        h = self.pointwise_conv(h)
        return h


class PointWiseConvTranspose1d(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointWiseConvTranspose1d, self).__init__()

        # Check required inputs
        check_required(self, {"in_channels":in_channels,"out_channels":out_channels})

        # Establish default inputs
        kwargs["groups"] = 1
        kwargs["kernel_size"] = 1
        kwargs["padding"] = 0

        # Declare operation
        self.pointwise_conv_transp = ConvTranspose1d(in_channels, out_channels, **kwargs)

        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.pointwise_conv_transp.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv_transp(x)


class DepthwiseConvTranspose1d(Module):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConvTranspose1d, self).__init__()
        
        # Check required inputs
        check_required(self, {"in_channels":in_channels, "kernel_size":kernel_size})

        # Establish default inputs
        kwargs["groups"] = in_channels
        kwargs["padding"] = kwargs.get("padding", (kernel_size-1)//2)
        if "out_channels" in kwargs:
            kwargs.pop("out_channels")
        
        # Declare operation
        self.depthwise_conv_transp = ConvTranspose1d(in_channels, in_channels, kernel_size, **kwargs)
        
        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.depthwise_conv_transp.weight)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.depthwise_conv_transp(x)


class SeparableConvTranspose1d(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConvTranspose1d, self).__init__()
        
        # Declare operations
        self.depthwise_conv_transp = DepthwiseConvTranspose1d(in_channels, kernel_size, **kwargs)
        self.pointwise_conv_transp = PointWiseConvTranspose1d(in_channels, out_channels, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        h = self.depthwise_conv_transp(x)
        h = self.pointwise_conv_transp(h)
        return h


class Dropout1d(Module):
    """Applies one-dimensional spatial dropout"""
    def __init__(self, p: [0., 1.], inplace: bool = False):
        super(Dropout1d, self).__init__()
        if (p < 0) or (p > 1):
            raise ValueError("Invalid probability {} provided. Must be formatted in range [0,1]".format(p))
        self.p = p
        self.inplace = inplace
        self.dropout = Dropout2d(self.p, self.inplace)
    
    def forward(self, x: Tensor) -> Tensor:
        # add a dimension for 2D to work -> format BxCxHxW
        x = x.unsqueeze(-1) 
        x = self.dropout(x).squeeze(-1)
        return x


class ImagePooling2d(Sequential):
    def __init__(self, in_channels: int = required, out_channels: int = required):
        super(ImagePooling2d, self).__init__()
        self.pooling = AdaptiveAvgPool2d(1)
        self.convolution = SeparableConv2d(in_channels, out_channels, 1, bias=False)
        self.batchnorm = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)

        # Check required inputs
        check_required(self, {"in_channels":in_channels, "out_channels":out_channels})

    def forward(self, x):
        size = x.shape[2:]
        x = self.pooling(x)
        x = self.convolution(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class PointWiseConv2d(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointWiseConv2d, self).__init__()

        # Check required inputs
        check_required(self, {"in_channels":in_channels, "out_channels":out_channels})

        # Establish default inputs
        kwargs["groups"] = 1
        kwargs["kernel_size"] = 1
        kwargs["padding"] = 0

        # Declare operation
        self.pointwise_conv = Conv2d(in_channels, out_channels, **kwargs)

        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.pointwise_conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(x)


class DepthwiseConv2d(Module):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConv2d, self).__init__()

        # Check required inputs
        check_required(self, {"in_channels":in_channels, "kernel_size":kernel_size})
        
        # Establish default inputs
        kwargs["groups"] = in_channels
        kwargs["padding"] = kwargs.get("padding", (kernel_size-1)//2)
        if "out_channels" in kwargs:
            kwargs.pop("out_channels")

        # Declare operation
        self.depthwise_conv = Conv2d(in_channels, in_channels, kernel_size, **kwargs)
        
        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.depthwise_conv.weight)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.depthwise_conv(x)


class SeparableConv2d(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConv2d, self).__init__()
        
        # Check required inputs
        check_required(self, {"in_channels":in_channels,"out_channels":out_channels,"kernel_size":kernel_size})

        # Declare operations
        self.depthwise_conv = DepthwiseConv2d(in_channels, kernel_size, **kwargs)
        self.pointwise_conv = PointWiseConv2d(in_channels, out_channels, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        h = self.depthwise_conv(x)
        h = self.pointwise_conv(h)
        return h


class PointWiseConvTranspose2d(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointWiseConvTranspose2d, self).__init__()

        # Check required inputs
        check_required(self, {"in_channels":in_channels,"out_channels":out_channels})

        # Establish default inputs
        kwargs["groups"] = 1
        kwargs["kernel_size"] = 1
        kwargs["padding"] = 0

        # Declare operation
        self.pointwise_conv_transp = ConvTranspose2d(in_channels, out_channels, **kwargs)

        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.pointwise_conv_transp.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv_transp(x)


class DepthwiseConvTranspose2d(Module):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConvTranspose2d, self).__init__()
        
        # Check required inputs
        check_required(self, {"in_channels":in_channels, "kernel_size":kernel_size})

        # Establish default inputs
        kwargs["groups"] = in_channels
        kwargs["padding"] = kwargs.get("padding", (kernel_size-1)//2)
        if "out_channels" in kwargs:
            kwargs.pop("out_channels")
        
        # Declare operation
        self.depthwise_conv_transp = ConvTranspose2d(in_channels, in_channels, kernel_size, **kwargs)
        
        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.depthwise_conv_transp.weight)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.depthwise_conv_transp(x)


class SeparableConvTranspose2d(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConvTranspose2d, self).__init__()
        
        # Declare operations
        self.depthwise_conv_transp = DepthwiseConvTranspose2d(in_channels, kernel_size, **kwargs)
        self.pointwise_conv_transp = PointWiseConvTranspose2d(in_channels, out_channels, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        h = self.depthwise_conv_transp(x)
        h = self.pointwise_conv_transp(h)
        return h


class SqueezeAndExcitationNd(Module):
    def __init__(self, channels: int = required, reduction_factor: int = required, dim: int = required):
        super(SqueezeAndExcitationNd, self).__init__()
        # Dimensions
        if dim == 1: self.pooling = AdaptiveAvgPool1d(1)
        if dim == 2: self.pooling = AdaptiveAvgPool2d(1)
        if dim == 3: self.pooling = AdaptiveAvgPool3d(1)
        if dim not in [1,2,3]: raise ValueError("'{}' dimensions not allowed".format(dim))

        # Rest of parameters
        self.encoder = Linear(channels, channels//reduction_factor, bias=False)
        self.relu = ReLU(inplace=True)
        self.decoder = Linear(channels//reduction_factor, channels, bias=False)
        self.sigmoid = Sigmoid()

        # Check required inputs
        check_required(self, {"channels":channels, "reduction_factor":reduction_factor})

    def forward(self, x):
        # Compute the squeeze tensor
        squeeze_tensor = self.pooling(x)
        squeeze_shape = squeeze_tensor.shape
        squeeze_tensor = squeeze_tensor.squeeze()
        squeeze_tensor = self.encoder(squeeze_tensor)
        squeeze_tensor = self.relu(squeeze_tensor)
        squeeze_tensor = self.decoder(squeeze_tensor)
        squeeze_tensor = self.sigmoid(squeeze_tensor)
        squeeze_tensor = squeeze_tensor.view(squeeze_shape)
        
        # Multiply by original tensor (will broadcast)
        return x*squeeze_tensor

class SqueezeAndExcitation1d(SqueezeAndExcitationNd):
    def __init__(self, channels: int = required, reduction_factor: int = required):
        super(SqueezeAndExcitation1d, self).__init__(channels, reduction_factor, dim=1)

class SqueezeAndExcitation2d(SqueezeAndExcitationNd):
    def __init__(self, channels: int = required, reduction_factor: int = required):
        super(SqueezeAndExcitation2d, self).__init__(channels, reduction_factor, dim=2)

class SqueezeAndExcitation3d(SqueezeAndExcitationNd):
    def __init__(self, channels: int = required, reduction_factor: int = required):
        super(SqueezeAndExcitation3d, self).__init__(channels, reduction_factor, dim=3)


class PointwiseSqueezeAndExcitationNd(Module):
    def __init__(self, channels: int = required, reduction_factor: int = required, dim: int = required):
        super(PointwiseSqueezeAndExcitationNd, self).__init__()
        if dim == 1: self.convolution = Conv1d(channels, 1, 1, bias=False)
        if dim == 2: self.convolution = Conv2d(channels, 1, 1, bias=False)
        if dim == 3: self.convolution = Conv3d(channels, 1, 1, bias=False)
        if dim not in [1,2,3]: raise ValueError("'{}' dimensions not allowed".format(dim))

        # Sigmoid operation
        self.sigmoid = Sigmoid()

        # Check required inputs
        check_required(self, {"channels":channels, "reduction_factor":reduction_factor})

    def forward(self, x):
        squeeze = self.convolution(x)
        squeeze = self.sigmoid(squeeze)
        return x*squeeze

class PointwiseSqueezeAndExcitation1d(PointwiseSqueezeAndExcitationNd):
    def __init__(self, channels: int = required, reduction_factor: int = required):
        super(PointwiseSqueezeAndExcitation1d, self).__init__(channels=channels, reduction_factor=reduction_factor, dim=1)

class PointwiseSqueezeAndExcitation2d(PointwiseSqueezeAndExcitationNd):
    def __init__(self, channels: int = required, reduction_factor: int = required):
        super(PointwiseSqueezeAndExcitation2d, self).__init__(channels=channels, reduction_factor=reduction_factor, dim=2)

class PointwiseSqueezeAndExcitation3d(PointwiseSqueezeAndExcitationNd):
    def __init__(self, channels: int = required, reduction_factor: int = required):
        super(PointwiseSqueezeAndExcitation3d, self).__init__(channels=channels, reduction_factor=reduction_factor, dim=3)


class EfficientChannelAttentionNd(Module):
    """Constructs a ECA module. Taken from "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    Args:
        channels: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """
    def __init__(self, channels: int = required, kernel_size: int = required, dim: int = required):
        super(EfficientChannelAttentionNd, self).__init__()
        if dim == 1: self.avg_pool = AdaptiveAvgPool1d(1)
        if dim == 2: self.avg_pool = AdaptiveAvgPool2d(1)
        if dim == 3: self.avg_pool = AdaptiveAvgPool3d(1)
        if dim not in [1,2,3]: raise ValueError("'{}' dimensions not allowed".format(dim))

        self.conv = Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = Sigmoid()

        check_required(self, {"channels": channels, "kernel_size": kernel_size, "dim": dim})

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y_shape = y.shape
        # Squeeze the tensor
        y = y.squeeze()[...,None]
        # Transpose the tensor
        y = y.transpose(-1,-2)
        # Two different branches of ECA module
        y = self.conv(y)
        # Transpose the tensor
        y = y.transpose(-1,-2)
        # Unsqueeze the tensor
        y = y.view(y_shape)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class EfficientChannelAttention1d(EfficientChannelAttentionNd):
    """Constructs a ECA module. Taken from "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    Args:
        channels: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """
    def __init__(self, channels, kernel_size=3):
        super(EfficientChannelAttention1d, self).__init__(channels=channels, kernel_size=kernel_size, dim=1)


class EfficientChannelAttention2d(EfficientChannelAttentionNd):
    """Constructs a ECA module. Taken from "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    Args:
        channels: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """
    def __init__(self, channels, kernel_size=3):
        super(EfficientChannelAttention2d, self).__init__(channels=channels, kernel_size=kernel_size, dim=2)


class EfficientChannelAttention3d(EfficientChannelAttentionNd):
    """Constructs a ECA module. Taken from "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    Args:
        channels: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """
    def __init__(self, channels, kernel_size=3):
        super(EfficientChannelAttention3d, self).__init__(channels=channels, kernel_size=kernel_size, dim=3)
