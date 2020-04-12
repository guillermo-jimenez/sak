from typing import Any, Tuple, List
from utils.__ops import required, check_required

import torch
import torch.nn
import utils
import utils.torch
import utils.torch.nn

def update_regularization(regularization_list: list = required, network_params: dict = required, preoperation=False):
    """Convenience function to update the values of different regularization strategies"""
    # Iterate over parameters list to check if update is needed
    for i in range(len(regularization_list)):
        # Iterate over contained parameters (exclude default)
        for arg in regularization_list[i].get('arguments',{}):
            # Case batchnorm
            if arg == 'num_features':
                # Check the model's operation parameters
                for p in network_params:
                    # If 
                    if preoperation and (p in ['in_channels', 'in_features', 'input_size', 'd_model']):
                        regularization_list[i]['arguments'][arg] = network_params[p]
                    elif not preoperation and (p in ['out_channels', 'out_features', 'hidden_size', 'd_model']):
                        regularization_list[i]['arguments'][arg] = network_params[p]

            # Keep adding
            pass
    return regularization_list
        
    
class CNN(torch.nn.Module):
    def __init__(self, 
                 channels: List[int] = required,
                 operation: dict = required,
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(CNN, self).__init__()
        
        # Store inputs
        self.channels = channels
        self.operation = utils.class_selector('utils.torch.nn',operation['name'])
        self.operation_params = operation['arguments']
        self.regularization = regularization
        self.regularize_extrema = regularize_extrema
        self.preoperation = preoperation
        
        # Create operations
        self.operations = []
        for i in range(len(channels)-1):
            # Update parameters
            self.operation_params['in_channels'] = channels[i]
            self.operation_params['out_channels'] = channels[i+1]
            
            # Update regularization parameters
            self.regularization = update_regularization(
                self.regularization,
                self.operation_params, 
                preoperation=self.preoperation)
            
            # If not preoperation, operation before regularization
            if self.preoperation:
                # Regularization
                if self.regularize_extrema or (not self.regularize_extrema and i != 0):
                    self.operations.append(utils.torch.nn.Regularization(self.regularization))
                    
                # Operation
                self.operations.append(self.operation(**self.operation_params))
            else:
                # Operation
                self.operations.append(self.operation(**self.operation_params))
                
                # Regularization
                if self.regularize_extrema or (not self.regularize_extrema and i != len(channels)-2):
                    self.operations.append(utils.torch.nn.Regularization(self.regularization))
                    
            
        # Create sequential model
        self.operations = torch.nn.Sequential(*self.operations)
    
    def forward(self, x: torch.Tensor) -> Any:
        return self.operations(x)


class DNN(torch.nn.Module):
    def __init__(self, 
                 features: List[int] = required,
                 operation: dict = required,
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(DNN, self).__init__()
        
        # Store inputs
        self.features = features
        self.operation = utils.class_selector('utils.torch.nn',operation['name'])
        self.operation_params = operation['arguments']
        self.regularization = regularization
        self.regularize_extrema = regularize_extrema
        self.preoperation = preoperation
        
        # Create operations
        self.operations = []
        for i in range(len(features)-1):
            # Update parameters
            self.operation_params['in_features'] = features[i]
            self.operation_params['out_features'] = features[i+1]
            
            # Update regularization parameters
            self.regularization = update_regularization(
                self.regularization,
                self.operation_params, 
                preoperation=self.preoperation)
            
            # If not preoperation, operation before regularization
            if self.preoperation:
                # Regularization
                if self.regularize_extrema or (not self.regularize_extrema and i != 0):
                    self.operations.append(utils.torch.nn.Regularization(self.regularization))
                    
                # Operation
                self.operations.append(self.operation(**self.operation_params))
            else:
                # Operation
                self.operations.append(self.operation(**self.operation_params))
                
                # Regularization
                if self.regularize_extrema or (not self.regularize_extrema and i != len(features)-2):
                    self.operations.append(utils.torch.nn.Regularization(self.regularization))
                    
            
        # Create sequential model
        self.operations = torch.nn.Sequential(*self.operations)
    
    def forward(self, x: torch.Tensor) -> Any:
        return self.operations(x)


