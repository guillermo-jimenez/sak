import torch
import torch.nn

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        pass

    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(torch.nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()
        pass

    def forward(self, x, shape):
        return x.view(shape)
        
class GlobalAvgPooling1d(torch.nn.Module):
    def __init__(self, dim: int = None, keepdims: bool = False):
        super(GlobalAvgPooling1d, self).__init__()

        self.dim = dim
        self.keepdims = keepdims

    def forward(self, x: torch.Tensor):
        return x.mean(self.dim, self.keepdims)
        
class PointWiseConv1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: dict):
        super(PointWiseConv1d, self).__init__()

        # Calculate padding and disable 'groups' argument
        if 'groups' in kwargs:  
            kwargs.pop('groups')
        if 'padding' in kwargs: 
            kwargs.pop('padding')

        # Declare operation
        self.pointwise_conv = torch.nn.Conv1d(in_channels, out_channels, padding=0, kernel_size=1, groups=1, **kwargs)

        # Initialize weights values
        torch.nn.init.xavier_normal_(self.pointwise_conv.weight)

    def forward(self, x: torch.Tensor):
        return self.pointwise_conv(x)

class DepthwiseConv1d(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, **kwargs: dict):
        super(DepthwiseConv1d, self).__init__()
        
        # Calculate padding and disable 'groups' argument
        if 'groups' in kwargs:  
            kwargs.pop('groups')
        if 'padding' in kwargs: 
            padding = kwargs['padding']
            kwargs.pop('padding')
        else:
            padding = (kernel_size-1)//2

        # Declare operation
        self.depthwise_conv = torch.nn.Conv1d(in_channels, in_channels,  padding=padding, kernel_size=kernel_size, groups=in_channels, **kwargs)
        
        # Initialize weights values
        torch.nn.init.xavier_normal_(self.depthwise_conv.weight)
        
    def forward(self, x: torch.Tensor):
        return self.depthwise_conv(x)

class SeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, **kwargs: dict):
        super(SeparableConv1d, self).__init__()
        
        self.depthwise_conv = DepthwiseConv1d(in_channels,kernel_size,**kwargs)
        self.pointwise_conv = PointWiseConv1d(in_channels,out_channels)
        
    def forward(self, x: torch.Tensor):
        h = self.depthwise_conv(x)
        h = self.pointwise_conv(h)
        return h

class PointWiseConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: dict):
        super(PointWiseConvTranspose1d, self).__init__()

        # Calculate padding and disable 'groups' argument
        if 'groups' in kwargs:  
            kwargs.pop('groups')
        if 'padding' in kwargs: 
            kwargs.pop('padding')

        # Declare operation
        self.pointwise_conv_transp = torch.nn.ConvTranspose1d(in_channels, out_channels, padding=0, kernel_size=1, groups=1, **kwargs)

        # Initialize weights values
        torch.nn.init.xavier_normal_(self.pointwise_conv_transp.weight)

    def forward(self, x: torch.Tensor):
        return self.pointwise_conv_transp(x)

class DepthwiseConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, **kwargs: dict):
        super(DepthwiseConvTranspose1d, self).__init__()
        
        # Calculate padding and disable 'groups' argument
        if 'groups' in kwargs:  
            kwargs.pop('groups')
        if 'padding' in kwargs: 
            padding = kwargs['padding']
            kwargs.pop('padding')
        else:
            padding = (kernel_size-1)//2

        # Declare operation
        self.depthwise_conv_transp = torch.nn.ConvTranspose1d(in_channels, in_channels,  padding=padding, kernel_size=kernel_size, groups=in_channels, **kwargs)
        
        # Initialize weights values
        torch.nn.init.xavier_normal_(self.depthwise_conv_transp.weight)
        
    def forward(self, x: torch.Tensor):
        return self.depthwise_conv_transp(x)

class SeparableConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, **kwargs: dict):
        super(SeparableConvTranspose1d, self).__init__()
        
        self.depthwise_conv_transp = DepthwiseConvTranspose1d(in_channels,kernel_size,**kwargs)
        self.pointwise_conv_transp = PointWiseConvTranspose1d(in_channels,out_channels)
        
    def forward(self, x: torch.Tensor):
        h = self.depthwise_conv_transp(x)
        h = self.pointwise_conv_transp(h)
        return h


