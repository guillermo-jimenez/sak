from typing import Dict
from sak import class_selector

class Mapper:
    def __init__(self, json: Dict):
        cls = class_selector(json["class"])
        self.function = cls(**json.get("arguments",{}))
        self.input_mappings = json["input_mappings"]
        self.output_mappings = json.get("output_mappings",[])
        
    def __call__(self, **kwargs):
        # Check input and output types
        assert all([isinstance(kwargs[k], Dict) for k in kwargs]), "Inputs and outputs must be specified as dicts"
        
        input_list = []
        for dict_from,element in self.input_mappings:
            input_list.append(kwargs[dict_from][element])
        
        output = self.function(*input_list)

        if len(self.output_mappings) == 0:
            return output
        else:
            for i,(dict_from,element) in enumerate(self.output_mappings):
                kwargs[dict_from][element] = output[i]
