# from . import functional_pil as F_pil
# from . import functional_tensor as F_t

# from torchvision.transforms.transforms import RandomTransforms

# torchvision.transforms.functional.adjust_brightness(img: torch.Tensor, brightness_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_contrast(img: torch.Tensor, contrast_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_saturation(img: torch.Tensor, saturation_factor: float) → torch.Tensor
# torchvision.transforms.functional.adjust_gamma(img, gamma, gain=1)

import torch
import numpy as np
import torchvision.transforms.functional as F
import random


class MyRandomFlip():
    """Vertically flip the Numpy arrays randomly with a given probability. Flip is performed on the last or second-to-last axis.

    Args:
        p (float): probability of the image being horizontally or vertically flipped, independently.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, im ,lbl):
        """
        Args:
            im_item,lbl_item are numpy arrays

        Returns:
            im_item,lbl_item: Randomly flipped arrays.
        """
        if torch.rand(1) < self.p:
            return np.flip(im, -1).copy(), np.flip(lbl, -1).copy()
        if torch.rand(1) < self.p:
            return np.flip(im, -2).copy(), np.flip(lbl, -2).copy()
        return im, lbl

    def __repr__(self):
        return "Numpy array horizontal/vertical flip " + '(p={})'.format(self.p)


class My_RandomGamma(torch.nn.Module):
    """Randomly adjust gamma of the different channels in sample['im'] independently. sample['lbl'] is unchanged.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        gamma (float):
    
    Todo:
        gamma_list: function will randomly choose gamma from a list of pre-specified values.
        Ensure independence of channels.
    """

    def __init__(self, p=0.5, gamma_list=[0.8,0.9,1.0,1.1,1.2]):
        super().__init__()
        self.p = p
        self.gamma_list = gamma_list
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
            G_im = F.adjust_gamma(G_im,random.choice(self.gamma_list),gain=1)
            R_im = F.adjust_gamma(R_im,random.choice(self.gamma_list),gain=1)
            G_im = F.pil_to_tensor(G_im)
            R_im = F.pil_to_tensor(R_im)
            return {'im':torch.cat([G_im,R_im],0),'lbl':sample['lbl']}
        return sample

    def __repr__(self):
        return self.__class__.__name__


class My_RandomContrast(torch.nn.Module):
    """Randomly adjust contrast of the different channels in sample['im'] independently. sample['lbl'] is unchanged.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        contrast_factor (float):  Non-negative float. 0 -> solid gray image, 1 -> original image, 2 increases contrast by factor of 2
    
    Todo:
        contrast_list: function will randomly choose contrast_factor from a list of pre-specified values.
        Ensure independence of channels.
    """

    def __init__(self, p=0.5, contrast_factor_list=[0.5,0.75,1.0,1.25,1.50]):
        super().__init__()
        self.p = p
        self.contrast_factor_list = contrast_factor_list
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
            G_im = F.adjust_contrast(G_im,contrast_factor=random.choice(self.contrast_factor_list))
            R_im = F.adjust_contrast(R_im,contrast_factor=random.choice(self.contrast_factor_list))
            G_im = F.pil_to_tensor(G_im)
            R_im = F.pil_to_tensor(R_im)
            return {'im':torch.cat([G_im,R_im],0),'lbl':sample['lbl']}
        return sample

    def __repr__(self):
        return self.__class__.__name__
