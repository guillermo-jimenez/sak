from typing import Any
from typing import List
from typing import Tuple
from typing import Callable

class Struct:
    def __init__(self, structure: dict = {}, **kwargs):
        structure.update(kwargs)
        for key, value in structure.items():
            setattr(self, key, value)
