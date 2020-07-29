from typing import Any, Tuple, List
import numpy as np
import pickle
import importlib

# Data loader to un-clutter code    
def load_data(filepath):
    dic = dict()
    with open(filepath) as f:
        text = list(f)
    for line in text:
        line = line.replace(' ','').replace('\n','').replace(',,','')
        if line[-1] == ',': line = line[:-1]
        head = line.split(',')[0]
        tail = line.split(',')[1:]
        if tail == ['']:
            tail = np.asarray([])
        else:
            tail = np.asarray(tail).astype(int)

        dic[head] = tail
    return dic

# Data loader to un-clutter code    
def save_data(filepath, dic):
    with open(filepath, 'w') as f:
        for key in dic.keys():
            # f.write("%s,%s\n"%(key,dic[key].tolist()))
            f.write("{},{}\n".format(key,str(dic[key].tolist()).replace(']','').replace('[','').replace(' ','')))

def map_upper(lst: list) -> list:
    return list(map(str.upper,lst))

def check_header(header):
    header = list(map(str.upper,header))
    header = np.array(header).squeeze()
    if header.ndim != 1:
        raise ValueError("Multi-dimensional header not allowed")
    if np.any((header[:,None] == np.unique(header)).sum(0) != 1):
        raise ValueError("Header with repeated entries not allowed")

def channel2index(channel: str, header=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']) -> int:
    if not isinstance(channel, str):
        raise ValueError("Channel '{}' type must be string. Provided type: {}.".format(channel,type(channel)))
    check_header(header)
    header = list(map(str.upper,header))
    location = np.where(np.array(header) == channel.upper())[0]
    if len(location) == 0:
        raise ValueError("Channel '{}' not found in header with channels {}".format(channel, header))
    return np.where(np.array(header) == channel.upper())[0][0]
    
def get_tqdm(iterator: iter, type: str = 'tqdm', **kwargs) -> Any:
    try:
        iterator = class_selector('tqdm', type)(iterator, **kwargs)
    except:
        iterator = iterator

    return iterator

def pickledump(obj: Any, file: str, mode: str = 'wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)

def pickleload(file: str, mode: str = 'rb') -> Any:
    with open(file, mode) as f:
        obj = pickle.load(f)
    return obj

def class_selector(module_name: str, class_name: str):
    # load the module, will raise ImportError if module cannot be loaded
    if class_name is 'None':
        class_name = class_name.lower()
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
            raise ValueError("Class {} instantiated without required parameter '{}'".format(obj.__class__, param))

required = _RequiredParameter()
