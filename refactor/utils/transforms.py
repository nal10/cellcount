#Collection of modified torchvision transforms
# import torch
# from torch import Tensor
# import math
# from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
# import numpy as np
# from numpy import sin, cos, tan
# import numbers
# from collections.abc import Sequence, Iterable
# import warnings

# from . import functional_pil as F_pil
# from . import functional_tensor as F_t

# from torchvision.transforms.transforms import RandomTransforms

# torchvision.transforms.functional.hflip(img: torch.Tensor) → torch.Tensor
# torchvision.transforms.functional.vflip(img: torch.Tensor) → torch.Tensor
# torchvision.transforms.functional.adjust_brightness(img: torch.Tensor, brightness_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_contrast(img: torch.Tensor, contrast_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_saturation(img: torch.Tensor, saturation_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_gamma(img, gamma, gain=1)

import torch
import torchvision.transforms.functional as F

class MyRandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        """
        Args:
            sample (dict): 'im' and 'lbl' are keys with corresponding values

        Returns:
            sample: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return {'im':F.hflip(sample['im']),'lbl':F.hflip(sample['lbl'])}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)