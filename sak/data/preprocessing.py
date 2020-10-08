from typing import Any, List, Tuple
import numpy as np

def label_group_split(y: list or np.ndarray, sep: str = "&", index: int = 0) -> Tuple[list, list]:
    y_label = []
    y_group = []
    
    for i in range(len(y)):
        code = y[i].split(sep)
        
        y_label.append(code[index])
        y_group.append(sep.join(code[:index] + code[index+1:]))
        
    return y_label, y_group
