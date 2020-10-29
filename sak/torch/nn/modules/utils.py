import sak
from typing import Any
from typing import List
from typing import Tuple
from torch import Tensor
from torch import Size
from torch import exp
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

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y


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

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
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


