import utils
import operator
import networkx
from collections import OrderedDict
from torch._jit_internal import _copy_to_script_wrapper
from itertools import islice
from typing import Any
from typing import List
from typing import Tuple
from numpy import array
from numpy import argsort
from numpy import arange
from numpy import nan
from torch import Tensor
from torch import Size
from torch import exp
from torch import randn_like
from torch.nn import Module
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
from torch.nn import Dropout
from torch.nn import Dropout2d
from torch.nn import Dropout3d
from torch.nn import AlphaDropout
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
from torch.nn import Transformer
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoderLayer
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Bilinear
from torch.nn import PixelShuffle
from torch.nn import Upsample
from torch.nn import UpsamplingNearest2d
from torch.nn import UpsamplingBilinear2d
from torch.nn import Flatten
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
from utils.__ops import required
from utils.__ops import check_required

"""
Order of operations
https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
"""

def update_regularization(regularization_list: list = required, network_params: dict = required, preoperation=False):
    """Convenience function to update the values of different regularization strategies"""
    # Iterate over parameters list to check if update is needed
    for i in range(len(regularization_list)):
        # Iterate over contained parameters (exclude default)
        for arg in regularization_list[i].get("arguments",{}):
            # Case batchnorm
            if arg == "num_features":
                # Check the model"s operation parameters
                for p in network_params:
                    # If 
                    if preoperation and (p in ["in_channels", "in_features", "input_size", "d_model"]):
                        regularization_list[i]["arguments"][arg] = network_params[p]
                    elif not preoperation and (p in ["out_channels", "out_features", "hidden_size", "d_model"]):
                        regularization_list[i]["arguments"][arg] = network_params[p]

            # Keep adding
            pass
    return regularization_list
        
    
class Lambda(Module):
    def __init__(self, lmbda, *args, **kwargs):
        super(Lambda, self).__init__()
        self.lmbda = lmbda
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        return self.lmbda(x, *self.args, **self.kwargs)
        
        
class ModelGraph(Module):
    r"""A model composer"""

    def __init__(self, json):
        super(ModelGraph, self).__init__()
        
        # Initialize operation graph
        self.graph = networkx.DiGraph()
        self.graph.add_node("input", returns=False)
        self.__return_order = []
        
        # Make space for plausible function names
        # Initialize with default function
        self.__function_list = [("forward","input")] 
        
        # Compose network
        self.__compose(json)
        
        # Order returns
        keys = [self.__return_order[i][0] for i in range(len(self.__return_order))]
        order = [self.__return_order[i][1] for i in range(len(self.__return_order))]
        keys = array(keys)[argsort(order)]
        self.__return_order = dict(zip(keys,arange(keys.size)))        
        
        # Initialize forward function
        for (fname, starting_node) in self.__function_list:
            setattr(self, fname, self.__set_fcn(fname,starting_node))
            
        
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        elif isinstance(idx, str):
            return self._modules[idx]
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Parallel, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())
        
    def __compose(self, structure, node_father = "input", acc_string = ""):
        # Marks start of a structure (series, parallel)
        if ("type" in structure) or ("modules" in structure):
            # If structure type is series:
            if structure.get("type","series").lower() == 'series':
                # Sanity check
                if "modules" not in structure:
                    raise ValueError("Missing module list of the structure")
                    
                # Add all elements as children of the previous element in the series
                for j in range(len(structure["modules"])):
                    res = self.__compose(structure["modules"][j], node_father, acc_string + "_" + str(j+1) if acc_string != "" else str(j+1))
                    if isinstance(res, tuple):
                        if j != len(structure["modules"])-1:
                            for r in res:
                                self.graph.add_edge(r, res)
                    else:
                        self.graph.add_edge(node_father, res)
                    node_father = res
            # If the structure is parallel (No default here, absorved by last conditional)
            elif structure["type"].lower() == 'parallel': 
                nodes = []
                for j in range(len(structure["modules"])):
                    res = self.__compose(structure["modules"][j], node_father, acc_string + "_" + str(j+1) if acc_string != "" else str(j+1))
                    nodes.append(res)
                    if res is not None:
                        self.graph.add_edge(node_father, res)
                return tuple(nodes)
            else:
                raise NotImplementedError(("Execution paths other than 'series' or 'parallel' " +
                                            "are not yet implemented. Inputted {}.".format(structure[0])))
        # Else is an operation
        else:
            # Sanity check: did not include tuples or lists
            if isinstance(structure, tuple) or isinstance(structure, list):
                raise ValueError("Beware, module contains nested list. Refer to API")
            
            # Add executable torch.nn.Module (must have "forward" function implemented) to pile
            self.add_module(structure.get('name',acc_string), utils.class_selector('utils.torch.nn',structure['module'])(**structure.get('arguments',{})))
            
            # Add module information to execution graph
            self.graph.add_node(structure.get('name',acc_string), returns=structure.get('returns',False))
            
            # Check if returns anything
            if structure.get('returns',False):
                self.__return_order.append((structure.get('name',acc_string),structure.get('order',nan)))
                
            # Check if it needs its own execution function
            if structure.get('function',False):
                self.__function_list.append((structure.get('function'),structure.get('name',acc_string)))
            
            # Return node name
            return structure.get('name',acc_string)

    # Function maker for all plausible addable functions
    def __set_fcn(self,name,start_node):
        # Retrieve function call with provided name. Static 
        # starting node (hence the difference between calls)
        def call(input: Tensor) -> Tuple[Tensor]:
            # Store input in partial computation
            partial = {start_node : input}
            
            # Check if "input" node is starting node (should be a 
            # better way with the graph, but little overhead anyway)
            is_start_node = False
            for node_from, nodes_to in self.graph.adjacency():
                # Check if it is starting node (should be a better
                # way with the graph, but little overhead anyway)
                if not is_start_node: 
                    if (node_from != start_node):
                        continue
                    else:
                        is_start_node = True
                
                # Retrieve input. Enclose in tuple to avoid splitting in tensor axes
                if isinstance(node_from, tuple):
                    x = []
                    for n in node_from:
                        x.append(partial[n])
                    x = tuple(x)
                else:
                    x = (partial[node_from],)

                # Compute output
                for n in nodes_to:
                    # If node_to is tuple, merging point of parallel
                    # branches (no operation to be performed in that case)
                    if not isinstance(n, tuple):
                        partial[n] = self[n](*x)
            
            # Declare outputs (vector of nan in case no return)
            output = [nan for _ in range(len(self.__return_order))]
            
            # Retrieve the nodes marked as outputs into the structure
            for n in partial:
                if self.graph.nodes[n]['returns']:
                    output[self.__return_order[n]] = partial[n]
            
            # Return output as a tuple
            return tuple(output)
        
        return call
        # setattr(self, name, call)
        
    def draw_networkx(self, ):
        try: # In case graphviz is installed (mostly for my own use)
            pos = networkx.drawing.nx_agraph.graphviz_layout(self.graph)
        except (NameError, ModuleNotFoundError, ImportError) as e:
            pos = networkx.drawing.layout.planar_layout(self.graph)

        networkx.draw_networkx_nodes(self.graph, pos,
                            nodelist=list(self.__return_order.keys()),
                            node_color='r')
        networkx.draw_networkx_nodes(self.graph, pos,
                            nodelist=[n for n in list(self.graph.nodes.keys()) if n not in list(self.__return_order.keys())],
                            node_color='b')
        networkx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
        networkx.draw_networkx_labels(self.graph, pos, dict(zip(self.graph.nodes.keys(),self.graph.nodes.keys())), font_size=16)


class Sequential(Module):
    r"""Another sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ("conv1", nn.Conv2d(1,20,5)),
                  ("relu1", nn.ReLU()),
                  ("conv2", nn.Conv2d(20,64,5)),
                  ("relu2", nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        elif isinstance(idx, str):
            return self._modules[idx]
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Parallel, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input):
        for module in self:
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


class Parallel(Module):
    r"""A parallel container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Parallel
        model = nn.Parallel(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Parallel with OrderedDict
        model = nn.Parallel(OrderedDict([
                  ("conv1", nn.Conv2d(1,20,5)),
                  ("relu1", nn.ReLU()),
                  ("conv2", nn.Conv2d(20,64,5)),
                  ("relu2", nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Parallel, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        elif isinstance(idx, str):
            return self._modules[idx]
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Parallel, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input):
        output = []
        for module in self:
            output.append(module(input))
        return tuple(output)


class CNN(Module):
    def __init__(self, 
                 channels: List[int] = required,
                 operation: dict = {"module" : "Conv1d"},
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(CNN, self).__init__()
        
        # Store inputs
        self.channels = channels
        self.operation = utils.class_selector("utils.torch.nn",operation["module"])
        self.operation_params = operation.get("arguments",{"kernel_size" : 3, "padding" : 1})
        self.regularization = regularization
        self.regularize_extrema = regularize_extrema
        self.preoperation = preoperation
        
        # Create operations
        self.operations = []
        for i in range(len(channels)-1):
            # Update parameters
            self.operation_params["in_channels"] = channels[i]
            self.operation_params["out_channels"] = channels[i+1]
            
            # Update regularization parameters
            if self.regularization:
                self.regularization = update_regularization(
                    self.regularization,
                    self.operation_params, 
                    preoperation=self.preoperation)
            
            # If not preoperation, operation before regularization
            if self.preoperation:
                # Regularization
                if self.regularization:
                    if self.regularize_extrema or (not self.regularize_extrema and i != 0):
                        self.operations.append(Regularization(self.regularization))
                    
                # Operation
                self.operations.append(self.operation(**self.operation_params))
            else:
                # Operation
                self.operations.append(self.operation(**self.operation_params))
                
                # Regularization
                if self.regularization:
                    if self.regularize_extrema or (not self.regularize_extrema and i != len(channels)-2):
                        self.operations.append(Regularization(self.regularization))
                    
            
        # Create sequential model
        self.operations = Sequential(*self.operations)
    
    def forward(self, x: Tensor) -> Any:
        return self.operations(x)


class DNN(Module):
    def __init__(self, 
                 features: List[int] = required,
                 operation: dict = {"module" : "Linear"},
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(DNN, self).__init__()
        
        # Store inputs
        self.features = features
        self.operation = utils.class_selector("utils.torch.nn",operation["module"])
        self.operation_params = operation.get("arguments",{})
        self.regularization = regularization
        self.regularize_extrema = regularize_extrema
        self.preoperation = preoperation
        
        # Create operations
        self.operations = []
        for i in range(len(features)-1):
            # Update parameters
            self.operation_params["in_features"] = features[i]
            self.operation_params["out_features"] = features[i+1]
            
            # Update regularization parameters
            if self.regularization:
                self.regularization = update_regularization(
                    self.regularization,
                    self.operation_params, 
                    preoperation=self.preoperation)
            
            # If not preoperation, operation before regularization
            if self.preoperation:
                # Regularization
                if self.regularization:
                    if self.regularize_extrema or (not self.regularize_extrema and i != 0):
                        self.operations.append(Regularization(self.regularization))
                    
                # Operation
                self.operations.append(self.operation(**self.operation_params))
            else:
                # Operation
                self.operations.append(self.operation(**self.operation_params))
                
                # Regularization
                if self.regularization:
                    if self.regularize_extrema or (not self.regularize_extrema and i != len(features)-2):
                        self.operations.append(Regularization(self.regularization))
                    
            
        # Create sequential model
        self.operations = Sequential(*self.operations)
    
    def forward(self, x: Tensor) -> Any:
        return self.operations(x)


class Regularization(Module):
    def __init__(self, operations: list):
        super(Regularization, self).__init__()
        self.operations = []
        for i in range(len(operations)):
            self.operations.append(utils.class_selector("utils.torch.nn",operations[i]["module"])(**operations[i].get("arguments",{})))
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
    """Does nothing apply dropout"""

    def __init__(self,*args,**kwargs):
        super(none, self).__init__()
        pass

    def forward(self, x):
        return x

class Identity(Module):
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

    def forward(self, x: Tensor) -> Tensor:
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


class GlobalAvgPooling1d(Module):
    def __init__(self, dim: int = None, keepdims: bool = False):
        super(GlobalAvgPooling1d, self).__init__()

        self.dim = dim
        self.keepdims = keepdims

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(self.dim, self.keepdims)


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
        initializer = utils.class_selector("torch.nn.init", kwargs.get("initializer","xavier_normal_"))
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
        initializer = utils.class_selector("torch.nn.init", kwargs.get("initializer","xavier_normal_"))
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
        initializer = utils.class_selector("torch.nn.init", kwargs.get("initializer","xavier_normal_"))
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
        initializer = utils.class_selector("torch.nn.init", kwargs.get("initializer","xavier_normal_"))
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


class Residual(Module):
    def __init__(self, in_channels: int = required, 
                       out_channels: int = required, 
                       kernel_size: int = required,
                       operation: str = required,
                       repetitions: int = required,
                       regularization: dict = {},
                      **kwargs: dict):
        super(Residual, self).__init__()
        
        # Check required inputs
        check_required(self, {"in_channels":in_channels,"out_channels":out_channels,"kernel_size":kernel_size,"operation":operation,"repetitions":repetitions})

        # Define operation to be performed
        self.repetitions = repetitions
        self.operation = utils.class_selector("utils.torch.nn", operation)

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
        self.operation_stack = Sequential(*self.operation_stack)

        # Operation if # of channels changes
        if __in_channels != out_channels:
            self.output_operation = self.operation(__in_channels, out_channels, kernel_size, **kwargs)

        # Residual
        self.addition = Add()

    def forward(self, x: Tensor) -> Tensor:
        h = x.clone()
        h = self.operation_stack(h)
        
        # If the number of channels of x and h does not coincide,
        # apply same transformation to x
        if x.shape[1] != h.shape[1]:
            x = self.output_operation(x)

        return self.addition(x, h) # Residual connection

class Dropout1d(Module):
    """Applies one-dimensional spatial dropout"""
    def __init__(self, p: [0., 1.]):
        super(Dropout1d, self).__init__()
        if (p < 0) or (p > 1):
            raise ValueError("Invalid probability {} provided. Must be formatted in range [0,1]".format(p))
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        # add a dimension for 2D to work -> format BxCxHxW
        x = x.unsqueeze(-1) 
        x = Dropout2d(self.p)(x).squeeze(-1)
        return x

