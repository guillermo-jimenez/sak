from typing import Union, Dict, List, Callable
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
        
    def __call__(self, **kwargs):
        # Check input and output types
        assert all([isinstance(kwargs[k], Dict) for k in kwargs]), "Inputs and outputs must be specified as dicts"
        
        input_args = []
        for dict_from,element in self.input_mappings:
            input_args.append(kwargs[dict_from][element])
        
        output = self.operation(*input_args)

        if len(self.output_mappings) == 0:
            return output
        else:
            if not isinstance(output, List):
                assert len(self.output_mappings) == 1, "Mismatch between length of resulting operation. Broadcasting..."
                output = [output]
            for i,(dict_from,element) in enumerate(self.output_mappings):
                kwargs[dict_from][element] = output[i]
