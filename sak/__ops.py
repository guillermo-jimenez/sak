from typing import Union, Dict, List, Callable, Iterable, Tuple, Any
import pandas as pd
import numpy as np
import os
import json
import copy
import pathlib
import datetime
import pickle
import importlib
import itertools
from warnings import warn

################ DEFINE REQUIRED BEFORE ANYTHING ELSE ################
# Blatantly stolen from https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

def check_required(obj, dic: dict):
    for (param, value) in dic.items():
        if value is required:
            raise ValueError("Class {} instantiated without required parameter {}".format(obj.__class__, param))

required = _RequiredParameter()

################ ANY OTHER THING ################


class Mapper:
    def __init__(self, operation: Union[Dict, Callable], input_mappings: List = [], output_mappings: Dict = []):
        if isinstance(operation, Callable):
            self.operation = operation
        elif isinstance(operation, Dict):
            self.operation = from_dict(operation)
        else:
            raise ValueError("Required input 'operation' provided with invalid type {}".format(type(operation)))
        self.input_mappings = input_mappings
        self.output_mappings = output_mappings
        
    def __call__(self, *args, **kwargs):
        # Check input and output types
        assert all([isinstance(kwargs[k], Dict) for k in kwargs]), "Inputs and outputs must be specified as dicts"
        
        input_args = []
        inputs = None
        for inputs in self.input_mappings:
            if isinstance(inputs, int):
                input_args.append(args[inputs])
            elif isinstance(inputs, List) or isinstance(inputs, Tuple):
                dict_from,element = inputs
                input_args.append(kwargs[dict_from][element])
            else:
                raise ValueError(f"""invalid input configuration, inputs should either be a list of indices (e.g. [0,8]) 
                or a nested list of dict keys (e.g. [['inputs', 'x_value'], ['outputs', 'y_value']]). Got {self.input_mappings} 
                as input mappings and the offending input pair is {inputs}""")
        
        output = self.operation(*input_args)
        mark_return = False

        if len(self.output_mappings) == 0:
            return output
        else:
            if (not isinstance(output, List)) and (not isinstance(output, Tuple)):
                assert len(self.output_mappings) == 1, "Mismatch between length of resulting operation. Broadcasting..."
                output = [output]
                
            for i,outputs in enumerate(self.output_mappings):
                if isinstance(outputs, int):
                    if isinstance(args, Iterable):
                        args = list(args)
                        mark_return = True
                    args[outputs] = output[i]
                elif isinstance(outputs, Iterable):
                    dict_from,element = outputs
                    kwargs[dict_from][element] = output[i]
                else:
                    raise ValueError(f"""invalid output configuration, outputs should either be a list of indices (e.g. [0,8]) 
                    or a nested list of dict keys (e.g. [['outputs', 'aaa'], ['inputs', 'bbb']]). Got {self.output_mappings} 
                    as output mappings and the offending output pair is {outputs}""")
        
        if mark_return:
            return tuple(args)


class Caller:
    def __init__(self, operation: Dict):
        print(operation)
        if isinstance(operation, Callable):
            self.operation = operation
        elif isinstance(operation, Dict):
            self.operation = class_selector(operation["class"])
        else:
            raise ValueError("Required input 'operation' provided with invalid type {}".format(type(operation)))
        self.arguments = arguments
        
    def __call__(self, *args, **kwargs):
        return self.operation(**self.arguments)


class ArgumentComposer:
    def __init__(self, operation: Dict, input_mappings: List = [], output_mappings: Dict = []):
        if isinstance(operation, Dict):
            self.operation = copy.deepcopy(operation)
        else:
            raise ValueError("Required input 'operation' provided with invalid type {}".format(type(operation)))
        self.input_mappings = input_mappings
        self.output_mappings = output_mappings
        
    def __call__(self, *args, **kwargs):
        # Check input and output types
        assert all([isinstance(kwargs[k], Dict) for k in kwargs]), "Inputs and outputs must be specified as dicts"
        
        # Retrieve operation
        operation = self.operation
        operation["arguments"] = operation.get("arguments", {})
        
        # Retrieve input arguments as dict
        for inputs in self.input_mappings:
            if isinstance(inputs, List) or isinstance(inputs, Tuple):
                if len(inputs) == 2:
                    dict_from,element = inputs
                    new_key_name = element
                elif len(inputs) == 3:
                    dict_from,element,new_key_name = inputs
                else:
                    raise ValueError("Only length 2 and length 3 lists are accepted (hacky)")
                operation["arguments"][new_key_name] = kwargs[dict_from][element]
            else:
                raise ValueError(f"""invalid input configuration, inputs should be a a nested list of dict keys 
                (e.g. [['inputs', 'x_value', 'rename_key'], ['outputs', 'y_value']]). Got 
                {self.input_mappings} as input mappings and the offending input pair is {inputs}""")
        
        # Generate operation
        output = from_dict(operation)
        mark_return = False

        if len(self.output_mappings) == 0:
            return output
        else:
            if (not isinstance(output, List)) and (not isinstance(output, Tuple)):
                assert len(self.output_mappings) == 1, "Mismatch between length of resulting operation. Broadcasting..."
                output = [output]
                
            for i,outputs in enumerate(self.output_mappings):
                if isinstance(outputs, int):
                    if isinstance(args, Iterable):
                        args = list(args)
                        mark_return = True
                    args[outputs] = output[i]
                elif isinstance(outputs, Iterable):
                    dict_from,element = outputs
                    kwargs[dict_from][element] = output[i]
                else:
                    raise ValueError(f"""invalid output configuration, outputs should either be a list of indices (e.g. [0,8]) 
                    or a nested list of dict keys (e.g. [['outputs', 'aaa'], ['inputs', 'bbb']]). Got {self.output_mappings} 
                    as output mappings and the offending output pair is {outputs}""")
        
        if mark_return:
            return tuple(args)


class SeedSetter:
    def __init__(self, seed: required):
        check_required(self,{"seed": seed})
        self.seed = seed
        self()
    
    def __call__(self):
        try:
            import numpy
            numpy.random.seed(self.seed)
        except ImportError:
            pass
        try:
            import random
            random.seed(self.seed)
        except ImportError:
            pass
        try:
            import torch
            torch.random.manual_seed(self.seed)
        except ImportError:
            pass

def to_tuple(*args):
    return tuple(args)

def set_seed(seed):
    try:
        import numpy
        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.random.manual_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow
        tensorflow.random.set_seed(seed)
    except ImportError:
        pass

def load_config(path: str, model_name: str = None, include_time: bool = True) -> dict:
    # Load json
    with open(path, "r") as f:
        config = json.load(f)

    # Get savedir string
    if "savedir" in config:          str_savedir = 'savedir'
    elif "save_directory" in config: str_savedir = 'save_directory'
    else: raise ValueError("Configuration file should include either the 'savedir' or 'save_directory' fields [case-sensitive]")

    # Expand user to avoid linux's ~ as alias to /home/$USER
    if "basedir" in config:
        config["basedir"] = os.path.expanduser(config["basedir"])

    if "datadir" in config:
        if isinstance(config["datadir"], str):
            config["datadir"] = os.path.expanduser(config["datadir"])
        elif isinstance(config["datadir"], (list, tuple)):
            for i in range(len(config["datadir"])):
                config["datadir"][i] = os.path.expanduser(config["datadir"][i])

    if str_savedir in config:
        config[str_savedir] = os.path.expanduser(config[str_savedir])

        # Add model name to savedir
        if model_name is None:
            # Split path
            root,file = os.path.split(path)
            model_name,ext = os.path.splitext(file)
        
        # Change path to contain model name
        if include_time:
            time = datetime.datetime.now().isoformat()
            time = time.replace(":","_").replace("-","_").replace(".","_").replace("T","___")
            config[str_savedir] = os.path.join(config[str_savedir], f"{model_name}_{time}")
        else:
            config[str_savedir] = os.path.join(config[str_savedir], model_name)

        # Make dir for output files
        if not os.path.isdir(config[str_savedir]):
            pathlib.Path(config[str_savedir]).mkdir(parents=True, exist_ok=True)

    return config

def invert_dict(dic: dict) -> dict:
    inv_dic = {}
    for k, v in dic.items():
        inv_dic[v] = inv_dic.get(v, []) + [k]
    return inv_dic

def as_tuple(*args: Tuple[Any]) -> Tuple:
    """Returns variable number of inputs as a tuple. Useful to load from json. Usage:
   
    >>> as_tuple(5, [2.3, "this"], min)
    (5, [2.3, 'this'], <built-in function min>)
    """
    return args

# Data loader to un-clutter code    
def load_data(file, dtype = int, start_dim = 1):
    dic = dict()
    with open(file) as f:
        text = list(f)
    for line in text:
        line = line.replace(" ","").replace("\n","").replace(",,","").replace("'","")
        if line[-1] == ",": line = line[:-1]
        head = ",".join(line.split(",")[:start_dim])
        tail = line.split(",")[start_dim:]
        if tail == [""]:
            tail = np.asarray([])
        else:
            tail = np.asarray(tail)
            if dtype is not None:
                tail = tail.astype(dtype)

        dic[head] = tail
    return dic

# Data saver to un-clutter code    
def save_data(dic,filepath):
    with open(filepath, 'w') as f:
        for key in dic.keys():
            # f.write("%s,%s\n"%(key,dic[key].tolist()))
            iterable = dic[key]
            if isinstance(iterable,pd.DataFrame):
                iterable = iterable.values
            if isinstance(iterable,np.ndarray):
                iterable = iterable.tolist()
            f.write("{},{}\n".format(key,str(iterable).replace(']','').replace('[','').replace(' ','')))

def map_upper(lst: list) -> list:
    return list(map(str.upper,lst))

def argsort_as(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    x = np.array(x)
    template = np.array(template)
    
    return np.argwhere(template[:,None] == x[None,:])[:,1]

def check_header(header):
    header = list(map(str.upper,header))
    header = np.array(header).squeeze()
    if header.ndim != 1:
        raise ValueError("Multi-dimensional header not allowed")
    if np.any((header[:,None] == np.unique(header)).sum(0) != 1):
        raise ValueError("Header with repeated entries not allowed")

def channel2index(channel: str, header=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]) -> int:
    if not isinstance(channel, str):
        raise ValueError("Channel {} type must be string. Provided type: {}.".format(channel,type(channel)))
    check_header(header)
    header = list(map(str.upper,header))
    location = np.where(np.array(header) == channel.upper())[0]
    if len(location) == 0:
        raise ValueError("Channel {} not found in header with channels {}".format(channel, header))
    return np.where(np.array(header) == channel.upper())[0][0]
    
def from_dict(operation: dict):
    # Check input
    assert isinstance(operation, dict), "Only nested dictionaries can be used as input"
    assert "class" in operation,        "Missing 'class' field in dictionary"

    # Default argument is empty
    operation["arguments"] = operation.get("arguments",{})
    break_nested = operation.get("break_nested",False)
    
    # If the argument field has nested calls
    if isinstance(operation["arguments"], dict) and not break_nested:
        for arg in operation["arguments"]:
            if isinstance(operation["arguments"][arg], dict):
                if ("class" in operation["arguments"][arg]):
                    operation["arguments"][arg] = from_dict(operation["arguments"][arg])
    if isinstance(operation["arguments"], list) and not break_nested:
        for i,arg in enumerate(operation["arguments"]):
            if isinstance(arg, dict):
                if ("class" in arg):
                    operation["arguments"][i] = from_dict(arg)
        
    if isinstance(operation["arguments"], list):
        return class_selector(operation["class"])(*operation["arguments"])
    elif isinstance(operation["arguments"], dict):
        return class_selector(operation["class"])(**operation["arguments"])
    else:
        return class_selector(operation["class"])(operation["arguments"])

def get_tqdm(iterator: iter, type: str = "tqdm.tqdm", **kwargs) -> Any:
    try:
        iterator = class_selector(type)(iterator, **kwargs)
    except:
        iterator = iterator

    return iterator

def pickledump(obj: Any, file: str, mode: str = "wb"):
    with open(file, mode) as f:
        pickle.dump(obj, f)

def pickleload(file: str, mode: str = "rb") -> Any:
    with open(file, mode) as f:
        obj = pickle.load(f)
    return obj

def class_selector(module_name: str, class_name: str = None):
    """Dynamically load module from a .json config file with the function call string.
    
    Parameters
    ----------
    module_name : str
        Name of the module to load. Might or might not contain the class name. 
        If it does contain the class name in the string, leave the class_name
        field as None
    
    class_name : str
        Name of the class within the module_name module. If the class name has
        been included in the module_name function, use None to mark empty.

    Examples
    --------

    >>> fnc = sak.class_selector('np.random.randint')
    >>> fnc(0,4)
    2
    >>> fnc = sak.class_selector('np.random','rand')
    >>> fnc()
    0.819275482191
    """
    # load the module, will raise ImportError if module cannot be loaded
    if class_name is None:
        class_name = module_name.split('.')[-1]
        module_name = '.'.join(module_name.split('.')[:-1])
    if class_name == "None":
        class_name = class_name.lower()
    if not module_name:
        module_name = "builtins"
    m = importlib.import_module(module_name)
        # return the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)

def reversed_enumerate(sequence, start=None):
    n = start
    if start is None:
        start = len(sequence) - 1
    for elem in sequence[::-1]:
        yield n, elem
        n -= 1    

def pairwise(iterator,n=2):
    iterators = itertools.tee(iterator,n)
    for i in range(len(iterators)):
        for j in range(i):
            next(iterators[i],None)
    return zip(*iterators)

def find_nested(dictionary, key, value):
    """https://stackoverflow.com/a/9808122/5211988"""
    for k, v in dictionary.items() if isinstance(dictionary, dict) else enumerate(dictionary) if isinstance(dictionary, list) else []:
        if (k == key) and (v == value):
            yield dictionary
        elif isinstance(v, dict):
            for result in find_nested(v, key,value):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find_nested(d, key, value):
                    yield result

def splitrfe(path: str) -> Tuple[str,str,str]:
    """Split a pathname. Returns tuple "(root, filename, extension)", where "filename" is
everything after the final slash and before the final dot and "extension" is everything after the final dot. Either part may be empty."""
    root,fname = os.path.split(path)
    fname,ext = os.path.splitext(fname)

    return root,fname,ext
