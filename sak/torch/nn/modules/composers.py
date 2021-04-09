from typing import Tuple, Union, Iterable, List, Dict, Optional, Callable
import sak
import operator
import networkx
import itertools
import numpy as np
import warnings
from collections import OrderedDict
from torch._jit_internal import _copy_to_script_wrapper
from torch import Tensor
from torch.nn import Module
from sak.__ops import required
from sak.__ops import check_required


class ModelGraph(Module):
    r"""A model composer"""

    def __init__(self, functions: List[Dict], nodes: List[Dict], edges: List[List]):
        super(ModelGraph, self).__init__()
        
        # Network's computational graph
        self.graph = networkx.DiGraph()

        # Retrieve output list for forward function
        self.__input_list = []
        self.__return_list = []
        for function in functions:
            if function['name'] == 'forward':
                self.__input_list += function['inputs']
                self.__return_list += function['outputs']

        # Add nodes to graph
        for node in nodes:
            is_node_input = node['id'] in self.__input_list
            does_node_return = node['id'] in self.__return_list
            self.graph.add_node(node['id'], returns=does_node_return, inputs=is_node_input)
            # Selected class
            cls = sak.class_selector(node['class'])
            # Add instantiated class to modules
            self.add_module(node['id'], cls(**node.get('arguments',{})))
        
        # Add edges to graph
        for (node_from,node_to) in edges:
            # Convert to tuples (list not acceptable as identifiers)
            if isinstance(node_from, list):
                node_from = tuple(node_from)
            if isinstance(node_to, list):
                node_to = tuple(node_to)

            # Add edges between nodes
            self.graph.add_edge(node_from, node_to)

        # Check structural integrity of the produced network
        terminal_outputs = set(itertools.chain(*[function['outputs'] for function in functions]))
        terminal_nodes = set([x for x in self.graph.nodes() if self.graph.out_degree(x)==0 and self.graph.in_degree(x)>=1])
        if len(terminal_nodes) != len(terminal_outputs):
            warnings.warn("Nodes {} are leafs but not marked as outputs. Check the provided config file".format(terminal_nodes-terminal_outputs))

        # Set output computational flows
        self.output_paths = {}
        self.subgraphs = {}
        for function in functions:
            # Retrieve paths
            all_executions = []
            subgraphs = []
            
            for output in function['outputs']:
                # 1. Compute all simple paths from all list of inputs to the specific output
                nodes_path = []
                for input in function['inputs']:
                    nodes_path.append(networkx.all_simple_paths(self.graph, input, output))
                
                # 2. Determine all necessary nodes for that specific output
                nodes_path = set(itertools.chain(*itertools.chain(*nodes_path)))

                # 2. Obtain subgraph
                subgraphs.append(networkx.DiGraph(self.graph.subgraph(nodes_path)))

                # 3. Order topologically the subset of nodes
                all_executions.append(list(networkx.topological_sort(subgraphs[-1])))

            self.subgraphs[function['name']] = subgraphs
            self.output_paths[function['name']] = all_executions
            
            # Set function
            self.compose(function['name'],function['inputs'],function['outputs'])

        
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(itertools.islice(iterator, idx, None))

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
        keys = super(ModelGraph, self).__dir__()
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
            end_nodes = np.array(end_nodes)
        else:
            end_nodes = np.array([end_nodes])
        
        # Retrieve execution information
        all_executions = self.output_paths[name]
        subgraphs = self.subgraphs[name]

        def call(input: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # Iterate over paths
            if not isinstance(input, Dict):
                raise ValueError("Only accepts a dict of torch tensors as input")
            # Assert all needed keys for the function are contained in the input array
            assert all([node in input for node in start_nodes]), "Check the call's input contains all the input tensors specified in the config file.\nProvided: {}\nNeeded: {}".format(list(input), list(start_nodes))
            
            # Define output structures
            partial = {}
            output = {}
            
            # Add inputs to "performed" computations
            partial.update(input)

            # Iterate over all network paths that lead to specific tensors marked as outputs
            # The "execution_order" variable is the one that carries the weight - has information
            # of the execution path from the input nodes to the output nodes ordered so that,
            # during the loop, you never need a partial result that has not been computed beforehand.
            for i,(execution_order,subgraph) in enumerate(zip(all_executions,subgraphs)):
                # Iterate over execution output path, from input node (no parents)
                for j,node_to in enumerate(execution_order):
                    # Check if node_to has already been computed (e.g. inputs)
                    if node_to in partial:
                        continue
                    
                    # Check all parent nodes, ordered topologically
                    nodes_from = tuple(subgraph.predecessors(node_to))
                    # If a single node is contained, mark to be computed
                    if len(nodes_from) == 1:
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
                    # If node_to is tuple, merging point of parallel/concatenation/etc
                    # branches (no operation to be performed in that case)
                    if not isinstance(node_to, tuple):
                        partial[node_to] = self[node_to](*x)

                # The last node computed is always a terminal node, and is added to the output list
                output[node_to] = partial[node_to]
            
            # Return output as a tuple
            return output

        # return call
        setattr(self, name, call)
        
    def draw_networkx(self, graph = None):
        # Prepare input
        if graph is None:
            graph = self.graph

        # In case graphviz is installed (mostly for my own use), use that layout
        pos = networkx.drawing.layout.planar_layout(graph)

        # Draw nodes/edges/labels            
        networkx.draw_networkx_nodes(graph, pos,
                            nodelist=self.__return_list,
                            node_color='r')
        networkx.draw_networkx_nodes(graph, pos,
                            nodelist=self.__input_list,
                            node_color='g')
        networkx.draw_networkx_nodes(graph, pos,
                            nodelist=[n for n in list(graph.nodes) if (n not in self.__return_list) and (n not in self.__input_list)],
                            node_color='b')
        networkx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
        networkx.draw_networkx_labels(graph, pos, dict(zip(list(graph.nodes.keys()),list(graph.nodes.keys()))), font_size=16)


class ModelWrapper(Module):
    def __init__(self, operation: Union[Dict, Callable], input_mappings: List, output_names: List):
        super(ModelWrapper, self).__init__()
        if isinstance(operation, Callable):
            self.operation = operation
        elif isinstance(operation, Dict):
            self.operation = from_dict(operation)
        else:
            raise ValueError("Required input 'operation' provided with invalid type {}".format(type(operation)))
        self.input_mappings = input_mappings
        self.output_names = output_names
        
    def forward(self, *args, **kwargs):
        if (len(args) == 1) and (len(kwargs) == 0):
            kwargs = {"inputs": args[0]}
            args = ()

        # Check input and output types
        assert all([isinstance(kwargs[k], Dict) for k in kwargs]), "Inputs and outputs must be specified as dicts"
        
        input_args = []
        for inputs in self.input_mappings:
            if isinstance(inputs, int):
                input_args.append(args[inputs])
            elif isinstance(inputs, List) or isinstance(inputs, Tuple):
                dict_from,element = inputs
                input_args.append(kwargs[dict_from][element])
        
        output = self.operation(*input_args)
        mark_return = False
        
        if isinstance(output, Tuple):
            if len(output) != len(self.output_names):
                raise ValueError(f"Mismatch between expected ({len(self.output_names)}) and obtained ({len(output)}) output variables")
            output = {self.output_names[i]: output[i] for i in range(len(self.output_names))}
        else:
            if len(self.output_names) != 1:
                raise ValueError(f"Mismatch between expected ({len(self.output_names)}) and obtained ({len(output)}) output variables")
            output = {self.output_names[0]: output for i in range(len(self.output_names))}

        return output   
    

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
        return next(itertools.islice(iterator, idx, None))

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
        keys = super(Sequential, self).__dir__()
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
        return next(itertools.islice(iterator, idx, None))

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


