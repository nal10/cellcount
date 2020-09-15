# from . import functional_pil as F_pil
# from . import functional_tensor as F_t

# from torchvision.transforms.transforms import RandomTransforms

# torchvision.transforms.functional.adjust_brightness(img: torch.Tensor, brightness_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_contrast(img: torch.Tensor, contrast_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_saturation(img: torch.Tensor, saturation_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_gamma(img, gamma, gain=1)

import torch
import torchvision.transforms.functional as F
import random


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
            sample: Randomly flipped sample dict.
        """
        if torch.rand(1) < self.p:
            return {'im':F.hflip(sample['im']),'lbl':F.hflip(sample['lbl'])}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class MyRandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given PIL Image randomly with a given probability.
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
            sample: Randomly flipped sample dict.
        """
        if torch.rand(1) < self.p:
            
            return {'im':F.vflip(sample['im']),'lbl':F.vflip(sample['lbl'])}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class My_RandomGamma(torch.nn.Module):
    """Randomly adjust gamma of the different channels in sample['im'] independently. sample['lbl'] is unchanged.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        gamma (float):
    
    Todo:
        gamma_list: function will randomly choose gamma from a list of pre-specified values.
    """

    def __init__(self, p=0.5, gamma=1.0):
        super().__init__()
        self.p = p
        self.gamma = gamma
        return

    def forward(self, sample):
        """
        Args:
            sample (dict): 'im' and 'lbl' are keys with corresponding values

        Returns:
            sample: Randomly saturated sample['im']. 
        """
        if torch.rand(1) < self.p:
            G_im = F.to_pil_image(sample['im'][0,:,:])
            R_im= F.to_pil_image(sample['im'][1,:,:])
            G_im = F.adjust_gamma(G_im,self.gamma,gain=1)
            R_im = F.adjust_gamma(R_im,self.gamma,gain=1)
            G_im = F.pil_to_tensor(G_im)
            R_im = F.pil_to_tensor(R_im)
            return {'im':torch.cat([G_im,R_im],0),'lbl':sample['lbl']}
        return sample

    def __repr__(self):
        return self.__class__.__name__