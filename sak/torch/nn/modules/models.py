from typing import Any, List, Tuple, Union, Dict
from torch import Tensor
from torch.nn import Module
from torch.nn import Identity
from .composers import Parallel
from .composers import Sequential
from .utils import Regularization
from sak import class_selector
from sak.__ops import required
from sak.__ops import check_required
from sak.__ops import from_dict
from sak.__ops import class_selector
from functools import reduce
from sak.torch.nn.modules.utils import Concatenate
from sak.__ops import reversed_enumerate


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
        self.operation_params = operation.get("arguments",{})
        if "convolution" in self.operation_params:
            for arg in self.operation_params["convolution"]:
                self.operation_params[arg] = self.operation_params["convolution"][arg]
        if "kernel_size" not in self.operation_params: self.operation_params["kernel_size"] = 3
        if     "padding" not in self.operation_params: self.operation_params["padding"] = 1
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


class UNet(Module):
    def __init__(self, 
                 in_channels: int = required,
                 out_channels: int = required,
                 unet_channels: int = required,
                 levels: int = required,
                 repetitions: int = required,
                 operation: dict = {"class" : "torch.nn.Conv1d"},
                 downsampling_operation: dict = {"class": "torch.nn.AvgPool1d", "arguments": {"kernel_size": 2}},
                 upsampling_operation: dict = {"class": "torch.nn.Upsample", "arguments": {"scale_factor": 2}},
                 merging_operation: dict = {"class": "sak.torch.nn.Concatenate"},
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 **kwargs):
        super(UNet, self).__init__()
        
        # Store inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_channels = unet_channels
        self.levels = levels
        self.repetitions = repetitions
        self.operation = operation
        self.downsampling_operation = downsampling_operation
        self.upsampling_operation = upsampling_operation
        self.merging_operation = merging_operation
        self.regularization = regularization
        self.regularize_extrema = regularize_extrema
        self.preoperation = preoperation
        
        # Infer scaling factor
        if   ("Upsample" in self.upsampling_operation["class"]) and ("scale_factor" in self.upsampling_operation.get("arguments")): 
            self.scale_factor = self.upsampling_operation["arguments"]["scale_factor"]
        elif ("AvgPool1d" in self.downsampling_operation["class"]) and ("kernel_size" in self.upsampling_operation.get("arguments")): 
            self.scale_factor = self.downsampling_operation["arguments"]["kernel_size"]
        elif ("AvgPool1d" in self.upsampling_operation["class"]) and ("kernel_size" in self.upsampling_operation.get("arguments")): 
            self.scale_factor = 1/self.upsampling_operation["arguments"]["kernel_size"]
        elif ("Upsample" in self.downsampling_operation["class"]) and ("scale_factor" in self.upsampling_operation.get("arguments")): 
            self.scale_factor = 1/self.downsampling_operation["arguments"]["scale_factor"]
            
        # Convolutional operation
        op = {
            "class": "sak.torch.nn.CNN",
            "arguments": {
                "operation": self.operation,
                "channels": [0]*self.repetitions,
                "regularization": self.regularization,
                "regularize_extrema": False,
                "preoperation": self.preoperation
            }
        }

        self.operations = []
        for i,level in enumerate(reversed(range(levels))):
            # Downsampling operation
            down_op = from_dict(self.downsampling_operation) if level != 0 else Identity()
            merg_op = from_dict(self.merging_operation)
            up_op   = from_dict(self.upsampling_operation) if level != 0 else Identity()
            
            # Define ENCODER channels operation
            encoder_op["arguments"]["channels"]     = [self.unet_channels*(self.scale_factor**level)]*self.repetitions
            encoder_op["arguments"]["channels"][0]  = self.in_channels if (level == 0) else int(encoder_op["arguments"]["channels"][0]/self.scale_factor)
            encoder_op["arguments"]["regularize_extrema"] = self.regularize_extrema if ((self.preoperation) or (not self.preoperation and (i != 0))) else False
            encoder_op = class_selector(encoder_op["class"])(**encoder_op["arguments"])

            # Define DECODER channels operation
            decoder_op["arguments"]["channels"]     = [self.unet_channels*(self.scale_factor**level)]*self.repetitions
            decoder_op["arguments"]["channels"][0]  = int(decoder_op["arguments"]["channels"][0] + decoder_op["arguments"]["channels"][0]*self.scale_factor)
            decoder_op["arguments"]["channels"][-1] = self.out_channels if (level == 0) else decoder_op["arguments"]["channels"][-1]
            decoder_op["arguments"]["regularize_extrema"] = self.regularize_extrema if ((not self.preoperation) or (self.preoperation and (i != 0))) else False
            decoder_op = class_selector(decoder_op["class"])(**decoder_op["arguments"]) if level == levels-1 else Identity()

            # Parallelize and sequentialize level
            if i == 0:
                level = Sequential(
                    down_op,
                    encoder_op,
                    up_op
                )
            elif i == levels-1:
                level = Sequential(
                    encoder_op,
                    Parallel(
                        Identity(),
                        self.operations
                    ),
                    Concatenate(),
                    decoder_op
                )
            else:
                level = Sequential(
                    down_op,
                    encoder_op,
                    Parallel(
                        Identity(),
                        self.operations
                    ),
                    Concatenate(),
                    decoder_op,
                    up_op
                )
            
            # Add operation to pile
            self.operations = level

        # Create sequential model
        self.operations = level


    
    def forward(self, x: Tensor) -> Any:
        return self.operations(x)


class DCC(Module):
    def __init__(self,
                 in_channels: int = required,
                 out_channels: int = required,
                 degree: Union[int, List[int]] = required,
                 operation: dict = {"class" : "torch.nn.Conv1d"},
                 regularization: list = None,
                 regularize_extrema: bool = True,
                 preoperation: bool = False,
                 include_input: bool = False,
                 **kwargs):
        super(DCC, self).__init__()
        
        # Fibonacci sequence
        fibonacci = lambda n: reduce(lambda x,n:[x[1],x[0]+x[1]], range(n),[0,1])[0]
        
        # Store inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.operation = class_selector(operation["class"])
        self.operation_params = operation.get("arguments",{})
        if "convolution" in self.operation_params:
            for arg in self.operation_params["convolution"]:
                self.operation_params[arg] = self.operation_params["convolution"][arg]
        if "kernel_size" not in self.operation_params: self.operation_params["kernel_size"] = 3
        if     "padding" not in self.operation_params: self.operation_params["padding"] = 1
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
            self.operation_params["padding"] = self.operation_params["dilation"]*(self.operation_params["kernel_size"]//2)

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
            self.operations.append(level_operations)
        
        # Sequential operations
        self.operations = Sequential(*self.operations)
            
        # Create sequential model
        self.merging_operation = Concatenate()
    
    def forward(self, x: Tensor) -> Any:
        out = [x]
        for op in self.operations:
            out.append(op(out[-1]))
        if not self.include_input:
            out = out[1:]
        out = self.merging_operation(*out)
        return out


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


class SelfAttention(Module):
    def __init__(self, convolution: dict = required, attention: dict = required, **kwargs):
        super(SelfAttention, self).__init__()
        # Map all input arguments to convolution's arguments
        convolution["arguments"] = convolution.get("arguments", {})
        for arg in ["in_channels","out_channels","kernel_size","padding",
                    "stride","dilation","groups","bias","padding_mode"]:
            if arg in kwargs:
                convolution["arguments"][arg] = kwargs[arg]
        
        # Try to get default arguments for attention
        attention["arguments"] = attention.get("arguments", {})
        if "out_channels" in convolution["arguments"]:
            attention["arguments"]["channels"] = convolution["arguments"]["out_channels"]
        elif "in_channels" in convolution["arguments"]:
            attention["arguments"]["channels"] = convolution["arguments"]["in_channels"]

        # Define operations
        self.convolution = class_selector(convolution["class"])(**convolution["arguments"])
        self.attention = class_selector(attention["class"])(**attention["arguments"])

    def forward(self, x):
        x = self.convolution(x)
        x = self.attention(x)
        return x

