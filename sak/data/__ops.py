from typing import Any
from typing import List
from typing import Tuple
from typing import Callable
import numpy as np
from sklearn.model_selection import train_test_split

class Struct:
    def __init__(self, structure: dict = {}, **kwargs):
        structure.update(kwargs)
        for key, value in structure.items():
            setattr(self, key, value)

def ball_scaling(X: np.ndarray, metric: Callable = lambda x: np.max(x)-np.min(x), radius: float = 1.0):
    """Balls of radius != 1 not implemented yet"""
    if radius != 1.0:
        raise NotImplementedError("Balls of radius != 1 not implemented yet")
    return X/(metric(X)+np.finfo(X.dtype).eps)


def split_train_valid_test(inputs: dict, valid_size: float = 1/4, test_size: float = 1/4) -> Tuple[dict,dict,dict]:
    # Get key list
    key_list = np.array(list(inputs["x"]))

    # Get indices of the keys
    ix_keys = np.arange(key_list.size)

    # Get sizes right over 100%
    valid_size = valid_size/(1-test_size)

    # If has groups, stratify
    if "group" in inputs:
        # Retrieve group in the same order as the key list
        groups = np.array([inputs["group"][k] for k in key_list])

        print(f"Dividing into groups with keys: {np.unique(groups)}")

        # Get train/valid/test keys
        ix_train,ix_test  = train_test_split( ix_keys,test_size=test_size,stratify=groups)
        ix_train,ix_valid = train_test_split(ix_train,test_size=valid_size,stratify=groups[ix_train])
    else:
        # Get train/valid/test keys
        ix_train,ix_test  = train_test_split( ix_keys,test_size=test_size)
        ix_train,ix_valid = train_test_split(ix_train,test_size=valid_size)

    # Generate new input sets
    inputs_train = {}
    inputs_valid = {}
    inputs_test  = {}

    for data in inputs:
        inputs_train[data] = {}
        inputs_valid[data] = {}
        inputs_test[data]  = {}

        for k in key_list[ix_train]:
            inputs_train[data][k] = inputs[data][k]
        for k in key_list[ix_valid]:
            inputs_valid[data][k] = inputs[data][k]
        for k in key_list[ix_test]:
            inputs_test[data][k] = inputs[data][k]

    # Sanity check
    msg = "Incorrect distribution of keys in train_test_split"
    k_train = np.array(list(inputs_train[data]))
    k_valid = np.array(list(inputs_valid[data]))
    k_test  = np.array(list(inputs_test[data]))
    
    assert (k_train[:,None] == k_valid[None,:]).sum() == 0, msg
    assert (k_train[:,None] == k_test[None,:]).sum() == 0, msg
    assert (k_valid[:,None] == k_test[None,:]).sum() == 0, msg
    
    return ((inputs_train,inputs_valid,inputs_test),(key_list,ix_train,ix_valid,ix_test))