from typing import Dict, Tuple, List
import torch
import torch.nn
import random
import torchvision.transforms
import numpy as np

from collections.abc import Sequence

# Check required arguments as keywords
from sak.__ops import from_dict
from sak.__ops import class_selector
from sak.__ops import required
from sak.__ops import check_required


class none(object):
    """Compose a random transform according to pre-specified config file"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args):
        return args


class AugmentationComposer(object):
    """Compose a random transform according to pre-specified config file"""

    def __init__(self, **dic):
        self.operations = self.__get_operation(dic)

    def __call__(self, *args: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        return self.operations(*args)

    def __get_operation(self, operation):
        if "transforms" in operation:
            ops = []
            for transform in operation["transforms"]:
                ops.append(self.__get_operation(transform))
            return class_selector(operation["class"])(ops)
        else:
            if isinstance(operation["arguments"], list):
                return class_selector(operation["class"])(*operation["arguments"])
            elif isinstance(operation["arguments"], dict):
                if "transforms" in operation["arguments"]:
                    operation["arguments"]["transforms"] = [from_dict(tr) for tr in operation["arguments"]["transforms"]]
                return class_selector(operation["class"])(**operation["arguments"])

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomTransforms:
    """Base class for a list of transformations with randomness

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (list or tuple or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5, per_element: bool = True):
        super().__init__()
        self.transforms = transforms
        self.p = p
        self.call = self.__per_element if self.per_element else self.__all

    def forward(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        return self.call(*x)

    def __per_element(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        raise NotImplementedError("Not yet implemented")

    def __all(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        if self.p < torch.rand(1):
            return x
        for t in self.transforms:
            x = t(*x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list. This transform does not support torchscript."""

    def __init__(self, transforms, p=None, per_element: bool = True):
        super().__init__(transforms)
        if p is not None and not isinstance(p, Sequence):
            raise TypeError("Argument p should be a sequence")
        self.per_element = per_element
        self.p = p
        self.call = self.__per_element if self.per_element else self.__all

    def __call__(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        return self.call(*x)

    def __per_element(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        if len(x) < 1:
            raise ValueError("Cannot take empty inputs")
        else:
            assert np.unique([t.shape[0] for t in x]).size == 1, "Batch sizes do not coincide, check inputs"
        bs = x[0].shape[0]
        tr = random.choices(self.transforms, weights=self.p, k=bs)

        y  = [torch.empty_like(t) for t in x]
        for i in range(bs):
            sub_x = tuple([x[j][i,None,] for j in range(len(x))])
            sub_out = tr[i](*sub_x)
            for j in range(len(x)):
                y[j][i] = sub_out[j][0]
        return tuple(y)

    def __all(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        t = random.choice(self.transforms)
        return t(*x)

    def __repr__(self):
        format_string = super().__repr__()
        format_string += f"(p={self.p})"
        return format_string


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order. This transform does not support torchscript."""

    def __init__(self, transforms, per_element: bool = True):
        super().__init__(transforms)
        self.per_element = per_element
        self.call = self.__per_element if self.per_element else self.__all

    def __call__(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        return self.call(*x)

    def __per_element(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        if len(x) < 1:
            raise ValueError("Cannot take empty inputs")
        else:
            assert np.unique([t.shape[0] for t in x]).size == 1, "Batch sizes do not coincide, check inputs"
        bs = x[0].shape[0]
        order = list(range(len(self.transforms)))
        y  = [torch.empty_like(t) for t in x]
        for i in range(bs):
            random.shuffle(order)
            sub_x = tuple([x[j][i,None,].clone() for j in range(len(x))])
            for j in order:
                # y[i] = (x[i,None,])[0]
                sub_x = self.transforms[j](*sub_x)
            # print(sub_x)
            for j in range(len(x)):
                y[j][i] = sub_x[j][0]

        return tuple(y)

    def __all(self, *x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        order = list(range(len(self.transforms)))
        for j in order:
            x = self.transforms[j](*x)
        return x
        

class CutMix:
    """Apply adapted CutMix (https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py)
    """
    def __init__(self, beta: float):
        self.beta = beta
        
    def __call__(self, *args: List[torch.Tensor]) -> List[torch.Tensor]:
        # Check inputs
        for i,elem_i in enumerate(args):
            assert isinstance(elem_i, torch.Tensor), "Non-tensor inputs provided"
            for j,elem_j in enumerate(args):
                if i == j: continue
                assert elem_i.shape[2:] == elem_j.shape[2:], "The shapes of the input tensors do not coincide"
        bs   = np.unique([v.shape[0] for v in args])
        ndim = np.unique([v.ndim     for v in args])
        assert   bs.size == 1, "The batch size is inconsistent"
        assert ndim.size == 1, "Number of dimensions are inconsistent"
        bs,ndim = bs[0],ndim[0]

        # If single batch size, nothing can be done
        if bs == 1:
            return args
        
        # Get beta value
        lmbda = np.random.beta(self.beta,self.beta)
        
        # Match source and destination
        match = np.random.permutation(np.arange(bs))
        zip_match = zip(match[:match.size//2],match[match.size//2:])

        # Iterate over matches
        outputs = []
        for i,elem in enumerate(args):
            outputs.append(elem.clone())
        for src,dst in zip_match:
            # Generate bounding box per match
            bbox = self.__rand_bbox(args[0].shape, lmbda)

            # Apply bounding box to all input tensors
            # TO DO - IMPROVE
            for i,elem in enumerate(args):
                if   len(bbox) == 1:
                    ((x1,y1),)                                  = bbox
                    tmp_src,tmp_dst                             = elem[src,:,x1:y1],elem[dst,:,x1:y1]
                    outputs[i][dst,:,x1:y1]                     = tmp_src
                    outputs[i][src,:,x1:y1]                     = tmp_dst
                elif len(bbox) == 2:
                    ((x1,y1),(x2,y2),)                          = bbox
                    tmp_src,tmp_dst                             = elem[src,:,x1:y1,x2:y2],elem[dst,:,x1:y1,x2:y2]
                    outputs[i][dst,:,x1:y1,x2:y2]               = tmp_src
                    outputs[i][src,:,x1:y1,x2:y2]               = tmp_dst
                elif len(bbox) == 3:
                    ((x1,y1),(x2,y2),(x3,y3),)                  = bbox
                    tmp_src,tmp_dst                             = elem[src,:,x1:y1,x2:y2,x3:y3],elem[dst,:,x1:y1,x2:y2,x3:y3]
                    outputs[i][dst,:,x1:y1,x2:y2,x3:y3]         = tmp_src
                    outputs[i][src,:,x1:y1,x2:y2,x3:y3]         = tmp_dst
                elif len(bbox) == 4:
                    ((x1,y1),(x2,y2),(x3,y3),(x4,y4))           = bbox
                    tmp_src,tmp_dst                             = elem[src,:,x1:y1,x2:y2,x3:y3,x4:y4],elem[dst,:,x1:y1,x2:y2,x3:y3,x4:y4]
                    outputs[i][dst,:,x1:y1,x2:y2,x3:y3,x4:y4]   = tmp_src
                    outputs[i][src,:,x1:y1,x2:y2,x3:y3,x4:y4]   = tmp_dst
                else: 
                    raise NotImplementedError("Not implemented for tensors of dimension larger than 4")

        return outputs

    def __rand_bbox(self, shape: List[int], lmbda: float) -> np.ndarray:
        bbox = []
        cut_ratio = np.sqrt(1. - lmbda)
        for size in shape[2:]:
            # Define cut size
            cut_size = int(size*cut_ratio)

            # Randomly define the location
            loc = np.random.randint(size)

            bbox.append((np.clip(loc-cut_size//2,0,size),np.clip(loc+cut_size//2,0,size)))

        return np.array(bbox)

