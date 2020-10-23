from typing import Any, List, Tuple, Union, Dict
from torch import Tensor
from torch.nn import Module
from .composers import Sequential
from .utils import Regularization
from sak import class_selector
from sak.__ops import required
from sak.__ops import check_required
from functools import reduce


def update_regularization(regularization_list: list = required, network_params: dict = required, preoperation=False):
    """Convenience function to update the values of different regularization strategies"""
    # Iterate over parameters list to check if update is needed
    for i,info in enumerate(regularization_list):
        # Iterate over contained parameters (exclude default)
        for arg in info.get("arguments",{}):
            # Case batchnorm
            if arg == "num_features":
                # Check the model"s operation parameters
                for p in network_params:
                    # If 
                    if preoperation and (p in ["in_channels", "in_features", "input_size", "d_model"]):
                        info["arguments"][arg] = network_params[p]
                    elif not preoperation and (p in ["out_channels", "out_features", "hidden_size", "d_model"]):
                        info["arguments"][arg] = network_params[p]

            # Keep adding
            pass
    return regularization_list
        
    
class CNN(Module):
    def __init__(self, 
                 channels: List[int] = required,
                 operation: dict = {"class" : "torch.nn.Conv1d"},
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(CNN, self).__init__()
        
        # Store inputs
        self.channels = channels
        self.operation = class_selector(operation["class"])
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
                 operation: dict = {"class" : "torch.nn.Linear"},
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(DNN, self).__init__()
        
        # Store inputs
        self.features = features
        self.operation = class_selector(operation["class"])
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


class DCN(Module):
    def __init__(self,
                 in_channels: int = required,
                 out_channels: int = required,
                 degree: Union[int, List[int]] = required,
                 operation: dict = {"class" : "torch.nn.Conv1d"},
                 regularization: list = None,
                 residual_operation: dict = {"class": "sak.torch.nn.Add"},
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(DCN, self).__init__()
        
        # Fibonacci sequence
        fibonacci = lambda n: reduce(lambda x,n:[x[1],x[0]+x[1]], range(n),[0,1])[0]
        
        # Store inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.operation = class_selector(operation["class"])
        self.operation_params = operation.get("arguments",{"kernel_size" : 3, "padding" : 1})
        self.residual_operation = residual_operation
        self.regularization = regularization
        self.regularize_extrema = regularize_extrema
        self.preoperation = preoperation
        if isinstance(degree, List):
            self.degree = degree
        else:
            self.degree = [fibonacci(d+1) for d in range(degree)]
        
        # Create operations
        self.operations = []
        for i in range(degree):
            level_operations = []
            # Update parameters
            self.operation_params["in_channels"] = self.in_channels if i == 0 else self.out_channels
            self.operation_params["out_channels"] = self.out_channels
            self.operation_params["dilation"] = self.degree[i]

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
                        level_operations.append(Regularization(self.regularization))
                    
                # Operation
                level_operations.append(self.operation(**self.operation_params))
            else:
                # Operation
                level_operations.append(self.operation(**self.operation_params))
                
                # Regularization
                if self.regularization:
                    if self.regularize_extrema or (not self.regularize_extrema and i != len(self.degree)-1):
                        level_operations.append(Regularization(self.regularization))
            
            # Convert to sequential
            level_operations = Sequential(*level_operations)
            
            # Apply residual operation
            if i != 0:
                self.operations.append(Residual(level_operations, self.residual_operation))
            else:
                self.operations.append(level_operations)
            
        # Create sequential model
        self.operations = Sequential(*self.operations)
    
    def forward(self, x: Tensor) -> Any:
        return self.operations(x)


class Residual(Module):
    def __init__(self, operation: Module = required,
                       residual_operation: dict = {"class": "sak.torch.nn.Add"},
                       output_operation: dict = {"class": "torch.nn.Conv1d"},
                       **kwargs: dict):
        super(Residual, self).__init__()
        
        # Check required inputs
        check_required(self, {"operation":operation})

        # Obtain inputs
        self.operation = operation
        self.residual_operation = class_selector(residual_operation["class"])(**residual_operation.get("arguments",{}))


    def forward(self, x: Tensor) -> Tensor:
        h = x.clone()
        h = self.operation(h)
        h = self.residual_operation(x,h)

        return h

