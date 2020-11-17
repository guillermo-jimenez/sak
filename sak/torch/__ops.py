from typing import Union, Dict, List, Callable, Iterable, Tuple
from sak import class_selector, from_dict
from warnings import warn

class Mapper:
    def __init__(self, operation: Union[Dict, Callable], input_mappings: List, output_mappings: Dict = []):
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
        for inputs in self.input_mappings:
            if isinstance(inputs, int):
                input_args.append(args[inputs])
            elif isinstance(inputs, List) or isinstance(inputs, Tuple):
                dict_from,element = inputs
                input_args.append(kwargs[dict_from][element])
        
        output = self.operation(*input_args)
        mark_return = False

        if len(self.output_mappings) == 0:
            return output
        else:
            if (not isinstance(output, List)) and (not isinstance(output, Tuple)):
                assert len(self.output_mappings) == 1, "Mismatch between length of resulting operation. Broadcasting..."
                output = [output]
                
            for i,outputs in enumerate(self.output_mappings):
                if isinstance(inputs, int):
                    if isinstance(args, Iterable):
                        args = list(args)
                        mark_return = True
                    args[outputs] = output[i]
                elif isinstance(inputs, Iterable):
                    dict_from,element = outputs
                    kwargs[dict_from][element] = output[i]
        
        if mark_return:
            return tuple(args)
