from typing import Any
import torch
import torch.nn
import utils
import utils.torch.nn

def compose(structure: Any) -> torch.nn.Module:
    if isinstance(structure, tuple):
        modules = []
        for i in range(len(structure)):
            modules.append(compose(structure[i]))
            
        modules = utils.torch.nn.Parallel(*modules)

    elif isinstance(structure, list):
        modules = []
        for i in range(len(structure)):
            modules.append(compose(structure[i]))
        modules = utils.torch.nn.Sequential(*modules)
    else:
        modules = utils.class_selector('utils.torch.nn',structure['name'])(**structure.get('arguments',{}))

    return modules
