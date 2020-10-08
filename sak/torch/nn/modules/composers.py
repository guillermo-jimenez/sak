import utils
import operator
import networkx
from collections import OrderedDict
from torch._jit_internal import _copy_to_script_wrapper
from itertools import islice
from typing import Tuple
from numpy import array
from numpy import argmax
from torch import Tensor
from torch import Size
from torch.nn import Module
from utils.__ops import required
from utils.__ops import check_required


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
            # Selected class
            cls = utils.class_selector(node['class'])
            # Add instantiated class to modules
            self.add_module(node['id'], cls(**node.get('arguments',{})))
        
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
            output = [None for _ in end_nodes]
            
            # Iterate over all assigned outputs
            for i,(execution_order,subgraph) in enumerate(zip(all_executions,subgraphs)):
                # Iterate over execution graph output node
                for j,node_to in enumerate(execution_order):
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


