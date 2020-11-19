from typing import Dict, Tuple, List
import torch
import torch.nn
import random
import torchvision.transforms

from collections.abc import Sequence

# Check required arguments as keywords
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

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, *img):
        if self.p < torch.rand(1):
            return img
        for t in self.transforms:
            img = t(*img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list. This transform does not support torchscript.
    """
    def __call__(self, *args):
        t = random.choice(self.transforms)
        return t(*args)


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order. This transform does not support torchscript.
    """
    def __call__(self, *img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](*img)
        return img


