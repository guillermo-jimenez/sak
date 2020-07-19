import utils
import operator
import networkx
from collections import OrderedDict
from torch._jit_internal import _copy_to_script_wrapper
from itertools import islice
from typing import Any
from typing import List
from typing import Tuple
from numpy import all
from numpy import array
from numpy import zeros
from numpy import arange
from numpy import argmax
from numpy import nan
from torch import Tensor
from torch import Size
from torch import exp
from torch import cat
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
from torch.nn.functional import interpolate
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
        
    
class ModelGraph(Module):
    r"""A model composer"""

    def __init__(self, json):
        super(ModelGraph, self).__init__()
        
        # Compose network
        self.graph = networkx.DiGraph()

        # Retrieve output list for forward function
        self.__return_list = []
        for function in json['functions']:
            if function['name'] == 'forward':
                self.__return_list += function['outputs']

        # Add nodes to graph
        for node in json['nodes']:
            does_node_return = node['id'] in self.__return_list
            self.graph.add_node(node['id'], returns=does_node_return)
            self.add_module(node['id'], utils.class_selector('utils.torch.nn',node['class'])(**node.get('arguments',{})))
        
        # Add edges to graph
        for edge_from, edge_to in json['edges']:
            # Convert to tuples (list not acceptable as identifiers)
            if isinstance(edge_from, list):
                edge_from = tuple(edge_from)
            if isinstance(edge_to, list):
                edge_to = tuple(edge_to)

            # Add edges between nodes
            self.graph.add_edge(edge_from, edge_to)

        # Set output computational flows
        self.output_paths = {}
        self.subgraphs = {}
        for function in json['functions']:
            # Retrieve paths
            all_executions = []
            subgraphs = []
            
            for output in function['outputs']:
                if len(function['inputs']) > 1:
                    raise NotImplementedError("Not yet implemented for more than one input per function")
                else:
                    input = function['inputs'][0]
                
                # 1. Determine all necessary nodes for that specific output
                nodes_path = set([n for l in list(networkx.all_simple_paths(self.graph,input,output)) for n in l])
                # 2. Obtain subgraph
                subgraphs.append(networkx.DiGraph(self.graph.subgraph(nodes_path)))
                # 3. Order topologically the subset of nodes
                all_executions.append(list(networkx.topological_sort(subgraphs[-1])))

            self.subgraphs[function['name']] = subgraphs
            self.output_paths[function['name']] = all_executions
            
            # Set function
            # setattr(self, function['name'], self.compose(function['name'],function['inputs'],function['outputs']))
            self.compose(function['name'],function['inputs'],function['outputs'])

        
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
        
    # Function maker for all plausible addable functions
    def compose(self,name,start_nodes,end_nodes):
        # Retrieve function call with provided name. Static 
        # starting node (hence the difference between calls)
        # List end nodes as array (for comparison)
        if isinstance(start_nodes, list):
            start_nodes = tuple(start_nodes)
        if isinstance(end_nodes,list):
            end_nodes = array(end_nodes)
        else:
            end_nodes = array([end_nodes])
        
        # Retrieve execution information
        all_executions = self.output_paths[name]
        subgraphs = self.subgraphs[name]

        def call(input: Tensor) -> Tuple[Tensor]:
            # Iterate over paths
            partial = {'input': input}
            output = [None for _ in range(len(end_nodes))]
            
            # Iterate over all assigned outputs
            for i in range(len(all_executions)):
                execution_order = all_executions[i]
                subgraph = subgraphs[i]

                # Iterate over execution graph output node
                for j in range(len(execution_order)):
                    node_to = execution_order[j]
                    # Check if node_to has already been computed
                    if node_to in partial:
                        continue
                    
                    # Check all nodes, ordered topologically
                    nodes_from = tuple(subgraph.predecessors(node_to))
                    if len(nodes_from) == 0:
                        nodes_from = 'input'
                    elif len(nodes_from) == 1:
                        nodes_from = nodes_from[0]

                    # Prepare inputs
                    if isinstance(nodes_from, tuple):
                        x = []
                        for n in nodes_from:
                            x.append(partial[n])
                        x = tuple(x)
                    else:
                        x = (partial[nodes_from],)

                    # Compute output
                    # If node_to is tuple, merging point of parallel
                    # branches (no operation to be performed in that case)
                    if not isinstance(node_to, tuple):
                        partial[node_to] = self[node_to](*x)

                # The last node computed is always a terminal node, and is added to the output list
                output[argmax(end_nodes == node_to)] = partial[node_to]
            
            # Return output as a tuple
            return tuple(output)
        # return call
        setattr(self, name, call)
        
    def draw_networkx(self, ):
        try: # In case graphviz is installed (mostly for my own use)
            pos = networkx.drawing.nx_agraph.graphviz_layout(self.graph)
        except (NameError, ModuleNotFoundError, ImportError) as e:
            pos = networkx.drawing.layout.planar_layout(self.graph)
            
        networkx.draw_networkx_nodes(self.graph, pos,
                            nodelist=self.__return_list,
                            node_color='r')
        networkx.draw_networkx_nodes(self.graph, pos,
                            nodelist=[n for n in list(self.graph.nodes) if n not in self.__return_list],
                            node_color='b')
        networkx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
        networkx.draw_networkx_labels(self.graph, pos, dict(zip(list(self.graph.nodes.keys()),list(self.graph.nodes.keys()))), font_size=16)


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
        
        
class CNN(Module):
    def __init__(self, 
                 channels: List[int] = required,
                 operation: dict = {"class" : "Conv1d"},
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(CNN, self).__init__()
        
        # Store inputs
        self.channels = channels
        self.operation = utils.class_selector("utils.torch.nn",operation["class"])
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
                 operation: dict = {"class" : "Linear"},
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(DNN, self).__init__()
        
        # Store inputs
        self.features = features
        self.operation = utils.class_selector("utils.torch.nn",operation["class"])
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
            self.operations.append(utils.class_selector("utils.torch.nn",operations[i]["class"])(**operations[i].get("arguments",{})))
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


class ImagePooling1d(Sequential):
    def __init__(self, in_channels: int = required, out_channels: int = required):
        super(ImagePooling1d, self).__init__(
            AdaptiveAvgPool1d(1),
            SeparableConv1d(in_channels, out_channels, 1, bias=False),
            BatchNorm1d(out_channels),
            ReLU(inplace=True))

        # Check required inputs
        check_required(self, {"in_channels":in_channels, "out_channels":out_channels})

    def forward(self, x):
        size = x.shape[2:]
        x = super(ImagePooling1d, self).forward(x)
        return interpolate(x.unsqueeze(-1), size=(*size,1), mode='bilinear', align_corners=False).squeeze(-1)
    

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


