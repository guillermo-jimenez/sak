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

class ImagePoolingNd(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, dim: int = required):
        super(ImagePoolingNd, self).__init__()
        # Check required inputs
        check_required(self, {"in_channels":in_channels, "out_channels":out_channels, "dim":dim})

        # Declare operations
        if dim == 1:
            self.pooling = AdaptiveAvgPool1d(1)
            self.convolution = SeparableConv1d(in_channels, out_channels, 1, bias=False)
            self.batchnorm = BatchNorm1d(out_channels)
            self.interpolation_mode = 'linear'
        elif dim == 2:
            self.pooling = AdaptiveAvgPool2d(1)
            self.convolution = SeparableConv2d(in_channels, out_channels, 1, bias=False)
            self.batchnorm = BatchNorm2d(out_channels)
            self.interpolation_mode = 'bilinear'
        elif dim == 3:
            self.pooling = AdaptiveAvgPool3d(1)
            self.convolution = SeparableConv3d(in_channels, out_channels, 1, bias=False)
            self.batchnorm = BatchNorm3d(out_channels)
            self.interpolation_mode = 'trilinear'
        else: 
            raise ValueError("Invalid number of dimensions: {}".format(dim))
            
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        x = self.pooling(x)
        x = self.convolution(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = interpolate(x, size=size, mode=self.interpolation_mode, align_corners=False)
        return x

class ImagePooling1d(ImagePoolingNd):
    def __init__(self, in_channels: int = required, out_channels: int = required):
        super(ImagePooling1d, self).__init__(in_channels, out_channels, dim=1)

class ImagePooling2d(ImagePoolingNd):
    def __init__(self, in_channels: int = required, out_channels: int = required):
        super(ImagePooling2d, self).__init__(in_channels, out_channels, dim=2)

class ImagePooling3d(ImagePoolingNd):
    def __init__(self, in_channels: int = required, out_channels: int = required):
        super(ImagePooling3d, self).__init__(in_channels, out_channels, dim=3)


class PointwiseConvNd(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, dim: int = required, **kwargs: dict):
        super(PointwiseConvNd, self).__init__()
        # Check required inputs
        check_required(self, {"in_channels":in_channels, "out_channels":out_channels, "dim":dim})

        # Establish default inputs
        kwargs["groups"] = tuple([1]*dim)
        kwargs["kernel_size"] = tuple([1]*dim)
        kwargs["padding"] = tuple([0]*dim)

        # Declare operations
        if   dim == 1: self.pointwise_conv = Conv1d(in_channels, out_channels, **kwargs)
        elif dim == 2: self.pointwise_conv = Conv2d(in_channels, out_channels, **kwargs)
        elif dim == 3: self.pointwise_conv = Conv3d(in_channels, out_channels, **kwargs)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))

        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.pointwise_conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(x)

class PointwiseConv1d(PointwiseConvNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointwiseConv1d, self).__init__(in_channels, out_channels, dim = 1, **kwargs)

class PointwiseConv2d(PointwiseConvNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointwiseConv2d, self).__init__(in_channels, out_channels, dim = 2, **kwargs)

class PointwiseConv3d(PointwiseConvNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointwiseConv3d, self).__init__(in_channels, out_channels, dim = 3, **kwargs)


class DepthwiseConvNd(Module):
    def __init__(self, in_channels: int = required, kernel_size: int = required, dim: int = required, **kwargs: dict):
        super(DepthwiseConvNd, self).__init__()
        # Check required inputs
        check_required(self, {"in_channels":in_channels, "kernel_size":kernel_size, "dim":dim})
        
        # Establish default inputs
        kwargs["groups"] = tuple([in_channels]*dim)
        kwargs["padding"] = kwargs.get("padding", tuple([(kernel_size-1)//2]*dim))
        if "out_channels" in kwargs:
            kwargs.pop("out_channels")

        # Declare operations
        if   dim == 1: self.depthwise_conv = Conv1d(in_channels, in_channels, kernel_size, **kwargs)
        elif dim == 2: self.depthwise_conv = Conv2d(in_channels, in_channels, kernel_size, **kwargs)
        elif dim == 3: self.depthwise_conv = Conv3d(in_channels, in_channels, kernel_size, **kwargs)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))

        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.depthwise_conv.weight)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.depthwise_conv(x)

class DepthwiseConv1d(DepthwiseConvNd):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConv1d, self).__init__(in_channels, kernel_size, dim = 1, **kwargs)

class DepthwiseConv2d(DepthwiseConvNd):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConv2d, self).__init__(in_channels, kernel_size, dim = 2, **kwargs)

class DepthwiseConv3d(DepthwiseConvNd):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConv3d, self).__init__(in_channels, kernel_size, dim = 3, **kwargs)


class SeparableConvNd(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, dim: int = required, **kwargs: dict):
        super(SeparableConvNd, self).__init__()
        # Check required inputs
        check_required(self, {"in_channels":in_channels,"out_channels":out_channels,"kernel_size":kernel_size,"dim":dim})

        # Declare operations
        if dim == 1:
            self.depthwise_conv = DepthwiseConv1d(in_channels, kernel_size, **kwargs)
            self.pointwise_conv = PointwiseConv1d(in_channels, out_channels, **kwargs)
        elif dim == 2:
            self.depthwise_conv = DepthwiseConv2d(in_channels, kernel_size, **kwargs)
            self.pointwise_conv = PointwiseConv2d(in_channels, out_channels, **kwargs)
        elif dim == 3:
            self.depthwise_conv = DepthwiseConv3d(in_channels, kernel_size, **kwargs)
            self.pointwise_conv = PointwiseConv3d(in_channels, out_channels, **kwargs)
        else: 
            raise ValueError("Invalid number of dimensions: {}".format(dim))
        
    def forward(self, x: Tensor) -> Tensor:
        h = self.depthwise_conv(x)
        h = self.pointwise_conv(h)
        return h

class SeparableConv1d(SeparableConvNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConv1d, self).__init__(in_channels, out_channels, kernel_size, dim = 1, **kwargs)

class SeparableConv2d(SeparableConvNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConv2d, self).__init__(in_channels, out_channels, kernel_size, dim = 2, **kwargs)

class SeparableConv3d(SeparableConvNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(SeparableConv3d, self).__init__(in_channels, out_channels, kernel_size, dim = 3, **kwargs)


class PointwiseConvTransposeNd(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, dim: int = required, **kwargs: dict):
        super(PointwiseConvTransposeNd, self).__init__()
        # Check required inputs
        check_required(self, {"in_channels":in_channels,"out_channels":out_channels,"dim":dim})

        # Establish default inputs
        kwargs["groups"] = tuple([1]*dim)
        kwargs["kernel_size"] = tuple([1]*dim)
        kwargs["padding"] = tuple([0]*dim)

        # Declare operations
        if   dim == 1: self.pointwise_conv_transp = ConvTranspose1d(in_channels, out_channels, **kwargs)
        elif dim == 2: self.pointwise_conv_transp = ConvTranspose2d(in_channels, out_channels, **kwargs)
        elif dim == 3: self.pointwise_conv_transp = ConvTranspose3d(in_channels, out_channels, **kwargs)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))

        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.pointwise_conv_transp.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv_transp(x)

class PointwiseConvTranspose1d(PointwiseConvTransposeNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointwiseConvTranspose1d, self).__init__(in_channels, out_channels, dim=1, **kwargs)

class PointwiseConvTranspose2d(PointwiseConvTransposeNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointwiseConvTranspose2d, self).__init__(in_channels, out_channels, dim=2, **kwargs)

class PointwiseConvTranspose3d(PointwiseConvTransposeNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, **kwargs: dict):
        super(PointwiseConvTranspose3d, self).__init__(in_channels, out_channels, dim=3, **kwargs)


class DepthwiseConvTransposeNd(Module):
    def __init__(self, in_channels: int = required, kernel_size: int = required, dim: int = required, **kwargs: dict):
        super(DepthwiseConvTransposeNd, self).__init__()
        # Check required inputs
        check_required(self, {"in_channels":in_channels, "kernel_size":kernel_size, "dim":dim})
        
        # Establish default inputs
        kwargs["groups"] = tuple([in_channels]*dim)
        kwargs["padding"] = kwargs.get("padding", tuple([(kernel_size-1)//2]*dim))
        if "out_channels" in kwargs:
            kwargs.pop("out_channels")

        # Declare operations
        if   dim == 1: self.depthwise_conv_transp = ConvTranspose1d(in_channels, in_channels, kernel_size, **kwargs)
        elif dim == 2: self.depthwise_conv_transp = ConvTranspose2d(in_channels, in_channels, kernel_size, **kwargs)
        elif dim == 3: self.depthwise_conv_transp = ConvTranspose3d(in_channels, in_channels, kernel_size, **kwargs)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))
        
        # Initialize weights values
        initializer = class_selector(kwargs.get("initializer","torch.nn.init.xavier_normal_"))
        initializer(self.depthwise_conv_transp.weight)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.depthwise_conv_transp(x)

class DepthwiseConvTranspose1d(DepthwiseConvTransposeNd):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConvTranspose1d, self).__init__(in_channels, kernel_size, dim = 1, **kwargs)

class DepthwiseConvTranspose2d(DepthwiseConvTransposeNd):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConvTranspose2d, self).__init__(in_channels, kernel_size, dim = 2, **kwargs)

class DepthwiseConvTranspose3d(DepthwiseConvTransposeNd):
    def __init__(self, in_channels: int = required, kernel_size: int = required, **kwargs: dict):
        super(DepthwiseConvTranspose3d, self).__init__(in_channels, kernel_size, dim = 3, **kwargs)


class SeparableConvTransposeNd(Module):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, dim: int = required, **kwargs: dict):
        super(SeparableConvTransposeNd, self).__init__()
        # Check required inputs
        check_required(self, {"in_channels":in_channels,"out_channels":out_channels,"kernel_size":kernel_size,"dim":dim})

        # Declare operations
        if dim == 1:
            self.depthwise_conv_transp = DepthwiseConvTranspose1d(in_channels, kernel_size, **kwargs)
            self.pointwise_conv_transp = PointwiseConvTranspose1d(in_channels, out_channels, **kwargs)
        elif dim == 2:
            self.depthwise_conv_transp = DepthwiseConvTranspose2d(in_channels, kernel_size, **kwargs)
            self.pointwise_conv_transp = PointwiseConvTranspose2d(in_channels, out_channels, **kwargs)
        elif dim == 3:
            self.depthwise_conv_transp = DepthwiseConvTranspose3d(in_channels, kernel_size, **kwargs)
            self.pointwise_conv_transp = PointwiseConvTranspose3d(in_channels, out_channels, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        h = self.depthwise_conv_transp(x)
        h = self.pointwise_conv_transp(h)
        return h

class SeparableConvTranspose1d(SeparableConvTransposeNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, dim: int = required, **kwargs: dict):
        super(SeparableConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, dim=1, **kwargs)

class SeparableConvTranspose2d(SeparableConvTransposeNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, dim: int = required, **kwargs: dict):
        super(SeparableConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, dim=2, **kwargs)
        
class SeparableConvTranspose3d(SeparableConvTransposeNd):
    def __init__(self, in_channels: int = required, out_channels: int = required, kernel_size: int = required, dim: int = required, **kwargs: dict):
        super(SeparableConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, dim=3, **kwargs)
        

class SqueezeAndExcitationNd(Module):
    def __init__(self, channels: int = required, reduction_factor: int = required, dim: int = required):
        super(SqueezeAndExcitationNd, self).__init__()
        # Check required inputs
        check_required(self, {"channels":channels, "reduction_factor":reduction_factor})

        # Declare operations
        if   dim == 1: self.pooling = AdaptiveAvgPool1d(1)
        elif dim == 2: self.pooling = AdaptiveAvgPool2d(1)
        elif dim == 3: self.pooling = AdaptiveAvgPool3d(1)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))

        self.encoder = Linear(channels, channels//reduction_factor, bias=False)
        self.relu = ReLU(inplace=True)
        self.decoder = Linear(channels//reduction_factor, channels, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Compute the squeeze tensor
        attention = self.pooling(x)
        shape = attention.shape
        attention = attention.squeeze()
        attention = self.encoder(attention)
        attention = self.relu(attention)
        attention = self.decoder(attention)
        attention = self.sigmoid(attention)
        attention = attention.view(shape)
        
        # Multiply by original tensor (will broadcast)
        return x*attention

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
        # Check required inputs
        check_required(self, {"channels":channels, "reduction_factor":reduction_factor, "dim":dim})

        # Declare operations
        if   dim == 1: self.convolution = Conv1d(channels, 1, 1, bias=False)
        elif dim == 2: self.convolution = Conv2d(channels, 1, 1, bias=False)
        elif dim == 3: self.convolution = Conv3d(channels, 1, 1, bias=False)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))

        # Sigmoid operation
        self.sigmoid = Sigmoid()

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
    def __init__(self, channels: int = required, kernel_size: int = required, dim: int = required, **kwargs: dict):
        super(EfficientChannelAttentionNd, self).__init__()
        # Check required inputs
        check_required(self, {"channels": channels, "kernel_size": kernel_size, "dim": dim})

        # Declare operations
        if   dim == 1: self.avg_pool = AdaptiveAvgPool1d(1)
        elif dim == 2: self.avg_pool = AdaptiveAvgPool2d(1)
        elif dim == 3: self.avg_pool = AdaptiveAvgPool3d(1)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))

        self.conv = Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = Sigmoid()


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


class AdaptiveAvgPoolAttentionNd(Module):
    def __init__(self, dim: int = required, **kwargs: dict):
        super(AdaptiveAvgPoolAttentionNd, self).__init__()
        # Check required inputs
        check_required(self, {"dim":dim})

        # Declare operations
        if   dim == 1: self.avgpool = AdaptiveAvgPool1d(1)
        elif dim == 2: self.avgpool = AdaptiveAvgPool2d(1)
        elif dim == 3: self.avgpool = AdaptiveAvgPool3d(1)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        pooled = self.avgpool(x)
        pooled = self.sigmoid(pooled)
        return x*pooled

class AdaptiveAvgPoolAttention1d(AdaptiveAvgPoolAttentionNd):
    def __init__(self, **kwargs: dict):
        super(AdaptiveAvgPoolAttention1d, self).__init__(dim = 1, **kwargs)

class AdaptiveAvgPoolAttention2d(AdaptiveAvgPoolAttentionNd):
    def __init__(self, **kwargs: dict):
        super(AdaptiveAvgPoolAttention2d, self).__init__(dim = 2, **kwargs)

class AdaptiveAvgPoolAttention3d(AdaptiveAvgPoolAttentionNd):
    def __init__(self, **kwargs: dict):
        super(AdaptiveAvgPoolAttention3d, self).__init__(dim = 3, **kwargs)


class PointwiseAttentionNd(Module):
    def __init__(self, channels: int = required, reduction_factor: int = required, dim: int = required, **kwargs: dict):
        super(PointwiseAttentionNd, self).__init__()
        # Check required inputs
        check_required(self, {"channels":channels,"reduction_factor":reduction_factor,"dim":dim})

        # Declare operations
        if   dim == 1: self.pooling = AdaptiveAvgPool1d(1)
        elif dim == 2: self.pooling = AdaptiveAvgPool2d(1)
        elif dim == 3: self.pooling = AdaptiveAvgPool3d(1)
        else: raise ValueError("Invalid number of dimensions: {}".format(dim))
        self.encoder = PointwiseConvNd(channels, max([1,channels/reduction_factor]), dim = dim, **kwargs)
        self.relu = ReLU(inplace=True)
        self.decoder = PointwiseConvNd(max([1,channels/reduction_factor]), channels, dim = dim, **kwargs)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Compute the squeeze tensor
        attention = self.pooling(x)
        attention = self.encoder(attention)
        attention = self.relu(attention)
        attention = self.decoder(attention)
        attention = self.sigmoid(attention)
        
        # Multiply by original tensor (will broadcast)
        return x*attention

class PointwiseAttention1d(PointwiseAttentionNd):
    def __init__(self, channels: int = required, reduction_factor: int = required, **kwargs: dict):
        super(PointwiseAttention1d, self).__init__(channels, reduction_factor, dim = 1, **kwargs)

class PointwiseAttention2d(PointwiseAttentionNd):
    def __init__(self, channels: int = required, reduction_factor: int = required, **kwargs: dict):
        super(PointwiseAttention2d, self).__init__(channels, reduction_factor, dim = 2, **kwargs)

class PointwiseAttention3d(PointwiseAttentionNd):
    def __init__(self, channels: int = required, reduction_factor: int = required, **kwargs: dict):
        super(PointwiseAttention3d, self).__init__(channels, reduction_factor, dim = 3, **kwargs)

