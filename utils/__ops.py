from typing import Any, Tuple, List
import pickle
import importlib

def get_tqdm(iterator: Any, type: str = 'tqdm', **kwargs) -> Any:
    try:
        iterator_class = class_selector('tqdm', type)(iterator, **kwargs)
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
            raise ValueError("Class {} instantiated without initializing parameter '{}'".format(obj.__class__, param))

required = _RequiredParameter()
