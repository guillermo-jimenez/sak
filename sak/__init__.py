# Import in correct order to avoid crash caused by torch:
# "ImportError: dlopen: cannot load any more object with static TLS"
import cv2
import sklearn
import skimage
import torch

from .__ops import as_tuple
from .__ops import channel2index
from .__ops import class_selector
from .__ops import pickleload
from .__ops import pickledump
from .__ops import invert_dict
from .__ops import map_upper
from .__ops import get_tqdm
from .__ops import from_dict
from .__ops import to_tuple
from .__ops import load_data
from .__ops import load_config
from .__ops import save_data
from .__ops import argsort_as
from .__ops import pairwise
from .__ops import reversed_enumerate
from .__ops import SeedSetter
from .__ops import Mapper
from .__ops import Caller
from .__ops import ArgumentComposer
from .__ops import splitrfe
from .__ops import find_nested

import sak.image
import sak.signal
import sak.visualization

__version__ = "0.0.2.37"