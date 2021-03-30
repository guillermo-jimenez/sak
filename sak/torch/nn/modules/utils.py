from typing import Tuple, List, Iterable, Union
from numbers import Number

import sak
from numpy import array
from typing import Any
from typing import List
from typing import Tuple
from torch import tensor
from torch import as_strided
from torch import Tensor
from torch import Size
from torch import exp
from torch import linspace
from torch import ones_like
from torch import sum
from torch import cat
from torch import randn_like
from torch.nn import Module
from torch.nn import Dropout2d
from .composers import Sequential
from sak.__ops import required
from sak.__ops import check_required

"""
Order of operations
https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
"""


class SoftArgmaxAlongAxis(Module):
    def __init__(self, axis: int, beta: float = 10):
        super(SoftArgmaxAlongAxis, self).__init__()
        self.axis = axis
        self.beta = beta
    
    def __call__(self, x: Tensor) -> Tensor:
        # Obtain per-pixel softmax of the input tensor alongside axis
        exponential = exp(self.beta*x)
        softmax = exponential/sum(exponential, axis=self.axis).unsqueeze(self.axis)

        # Obtain directional ramp
        directional = ones_like(x)*linspace(0,1,x.shape[self.axis],dtype=x.dtype)[:,None].to(x.device)
        
        # Obtain the argmax alongside axis
        argmax = sum(softmax*directional, axis=self.axis)*x.shape[self.axis]
        
        return argmax


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


class Lambda(Module):
    def __init__(self, lmbda, *args, **kwargs):
        super(Lambda, self).__init__()
        self.lmbda = lmbda
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        return self.lmbda(x, *self.args, **self.kwargs)
        
        
class Concatenate(Module):
    def __init__(self, dim: int = 1, *args, **kwargs):
        super(Concatenate, self).__init__()
        self.dim = dim
        pass

    def forward(self, *x_list: List[Tensor]) -> Tensor:
        return cat(x_list, dim=self.dim)
        

class PrintShapes(Module):
    def __init__(self, *args, **kwargs):
        super(PrintShapes, self).__init__()
        pass

    def forward(self, *x_list: List[Tensor]) -> Tensor:
        for x in x_list:
            print(x.shape)
        print("")

class OperationIterator(Module):
    def __init__(self, operation: dict = required, iterator: dict = required):
        super(OperationIterator, self).__init__()
        # Check required inputs
        check_required(self, {"operation": operation, "iterator": iterator})
        
        self.operation = sak.from_dict(operation)
        self.iterator = sak.from_dict(iterator)
        
    def forward(self, x: Tensor) -> List[Tensor]:
        output = [self.operation(xhat) for xhat in self.iterator(x)]
        return output


class ViewAsWindows(Module):
    """Inspired/partly copied from skimage.util.view_as_windows. 
    Returns unexpensive view of the tensor for iterative purposes"""
    def __init__(self, window_shape: Tuple = required, step: int = 1):
        super(ViewAsWindows, self).__init__()
        if isinstance(window_shape, Number):
            self.window_shape = (window_shape,)
        else:
            self.window_shape = window_shape
        self.step = step
    
    def forward(self, x: Tensor) -> List[Tensor]:
        # -- basic checks on arguments
        if not isinstance(x, Tensor):
            raise TypeError("`x` must be a torch tensor")

        ndim = x.ndim
        window_shape = self.window_shape
        step = int(self.step)
        
        if isinstance(window_shape, Number):
            window_shape = (window_shape,) * ndim
        if not (len(window_shape) == ndim):
            window_shape = tuple([x.shape[0],x.shape[1]] + [ws for ws in window_shape])

            if not (len(window_shape) == ndim):
                raise ValueError("`window_shape` is incompatible with `x.shape`")

        if isinstance(step, Number):
            if step < 1:
                raise ValueError("`step` must be >= 1")
            step = tuple([1,1] + [step] * (ndim-2))
        if len(step) != ndim:
            raise ValueError("`step` is incompatible with `x.shape`")

        arr_shape = array(x.shape)
        window_shape = array(window_shape, dtype=int)

        if ((arr_shape - window_shape) < 0).any():
            raise ValueError("`window_shape` is too large")

        if ((window_shape - 1) < 0).any():
            raise ValueError("`window_shape` is too small")

        # -- build rolling window view
        slices = tuple(slice(None, None, st) for st in step)
        window_strides = array(x.stride())

        indexing_strides = x[slices].stride()[2:]

        win_indices_shape = (((array(x.shape) - array(window_shape))
                              // array(step)) + 1)[2:]

        new_shape = tuple(list(win_indices_shape) + list(window_shape))
        strides = tuple(list(indexing_strides) + list(window_strides))

        y = as_strided(x, size=new_shape, stride=strides)
        return y
        

class ViewDimensionAsWindows(Module):
    """Inspired/partly copied from skimage.util.view_as_windows. 
    Returns unexpensive view of the tensor for iterative purposes"""
    def __init__(self, window_shape: int = required, dim: int = -1, step: int = 1):
        super(ViewDimensionAsWindows, self).__init__()
        self.window_shape = window_shape
        self.step = step
        self.dim = dim
        
        assert isinstance(window_shape, int), f"The provided value for input window_shape is invalid. Provided {type(window_shape)}; required int"
        assert isinstance(dim, int), f"The provided value for input dim is invalid. Provided {type(dim)}; required int"
        assert isinstance(step, int), f"The provided value for input step is invalid. Provided {type(step)}; required int"
    
    def forward(self, x: Tensor) -> List[Tensor]:
        # -- basic checks on arguments
        if not isinstance(x, Tensor):
            raise TypeError("`x` must be a torch tensor")

        ndim = x.ndim
        dim = self.dim
        
        # Determine window shape
        window_shape = list(x.shape)
        window_shape[dim] = self.window_shape

        # Determine step size
        step = [1]*(ndim)
        step[dim] = self.step

        arr_shape = array(x.shape)
        window_shape = array(window_shape, dtype=int)

        if ((arr_shape - window_shape) < 0).any():
            raise ValueError("`window_shape` is too large")

        if ((window_shape - 1) < 0).any():
            raise ValueError("`window_shape` is too small")

        # -- build rolling window view
        slices = tuple(slice(None, None, st) for st in step)
        window_strides = array(x.stride())

        indexing_strides = [x[slices].stride()[dim]]

        win_indices_shape = [(((array(x.shape) - array(window_shape))
                                  // array(step)) + 1)[dim]]
        new_shape = tuple(list(win_indices_shape) + list(window_shape))
        strides = tuple(list(indexing_strides) + list(window_strides))

        y = as_strided(x, size=new_shape, stride=strides)
        return y
        

class Regularization(Module):
    def __init__(self, operations: list):
        super(Regularization, self).__init__()
        self.operations = []
        for i in range(len(operations)):
            self.operations.append(sak.class_selector(operations[i]["class"])(**operations[i].get("arguments",{})))
        self.operations = Sequential(*self.operations)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.operations(x)


class Reparameterize(Module):
    def __init__(self, *args, **kwargs):
        super(Reparameterize, self).__init__()
        pass
    
    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = exp(0.5*logvar)
        eps = randn_like(std)
        return mu + eps*std


class none(Module):
    """Does nothing"""

    def __init__(self,*args,**kwargs):
        super(none, self).__init__()
        pass

    def forward(self, x):
        return x


class Add(Module):
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

    def forward(self, x: Union[Tensor, List, Tuple], y: Tensor = None) -> Tensor:
        if y is None:
            if isinstance(x, List) or isinstance(x, Tuple):
                out = 0
                for xhat in x:
                    out += xhat
                return out
            else:
                raise ValueError("Incorrect arguments")
        else:
            return x * y



class Multiply(Module):
    r"""A placeholder identity operator for multiplication.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = Multiply(54, unused_argument1=0.1, unused_argument2=False)
        >>> x = torch.randn(128, 20)
        >>> y = torch.randn(128, 20)
        >>> output = m(x,y)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Multiply, self).__init__()

    def forward(self, x: Union[Tensor, List, Tuple], y: Tensor = None) -> Tensor:
        if y is None:
            if isinstance(x, List) or isinstance(x, Tuple):
                out = 1
                for xhat in x:
                    out *= xhat
                return out
            else:
                raise ValueError("Incorrect arguments")
        else:
            return x * y


class Squeeze(Module):
    def __init__(self, *args, **kwargs):
        super(Squeeze, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.squeeze()


class Unsqueeze(Module):
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
    def __init__(self, dim, *args, **kwargs):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(self.dim)


class View(Module):
    def __init__(self, *shape, **kwargs):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(*self.shape)


class UnFlatten(Module):
    def __init__(self, shape: Size or list or tuple):
        super(UnFlatten, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], *self.shape)


class Interpolate(Module):
    def __init__(self, 
            size: (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]) = None, 
            scale_factor: (float or Tuple[float]) = None, 
            mode: ['nearest' or 'linear' or 'bilinear' or 'bicubic' or 'trilinear' or 'area'] = 'nearest', 
            align_corners: bool = None, 
            recompute_scale_factor: bool = None
        ):
        super(Interpolate, self).__init__()

        size = size
        scale_factor = scale_factor
        mode = mode
        align_corners = align_corners
        recompute_scale_factor = recompute_scale_factor

    def forward(self, input):
        return interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners, self.recompute_scale_factor)

