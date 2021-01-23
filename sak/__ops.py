from typing import Any, Tuple, List
import pandas as pd
import numpy as np
import pickle
import importlib


def invert_dict(dic: dict) -> dict:
    inv_map = {}
    for k, v in dic.items():
        inv_map[v] = dic.get(v, []) + [k]
    return inv_map

def as_tuple(*args: Tuple[Any]) -> Tuple:
    """Returns variable number of inputs as a tuple. Useful to load from json. Usage:
   
    >>> as_tuple(5, [2.3, "this"], min)
    (5, [2.3, 'this'], <built-in function min>)
    """
    return args

# Data loader to un-clutter code    
def load_data(file, dtype=int,start_dim=1):
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
    
    # If the argument field has nested calls
    if isinstance(operation["arguments"], dict):
        for arg in operation["arguments"]:
            if isinstance(operation["arguments"][arg], dict):
                if ("class" in operation["arguments"][arg]):
                    operation["arguments"][arg] = from_dict(operation["arguments"][arg])
    if isinstance(operation["arguments"], list):
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
        print("valid tqdm not found, inputted: {}. continuing...".format(iterator))
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
