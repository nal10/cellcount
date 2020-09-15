import glob
from aicsimageio import imread
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler

class ai224_RG(Dataset):
    """Dataset for training images

    Args:
        im_path: directory path with image .tif files 
        lbl_path: directory path with label .tif files 
        pad: padding added to im and lbl along the x and y axis
        patch_size: patch_size of returned items
    """

    def __init__(self,
                 pad=130,
                 patch_size = 260,
                 im_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Unet_tiles_082020/',
                 lbl_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Unet_tiles_082020/'):
        

        super().__init__()
        file_list = glob.glob(im_path+'/*_green.tif')
        file_list = glob.glob(im_path+'*_green.tif')
        file_list = [f.split('green.tif')[0] for f in file_list]
        file_list = [f.split(im_path)[1] for f in file_list]

        IM_list = []
        lbl_list = []
        
        labels=[0,1,2]
        one_hot = lambda x:np.concatenate([x==l for l in labels],axis=0)
        
        for f in file_list:
            #G_IM, G_lbl etc. are shape=(1,x,y)
            get_arr = lambda f: np.expand_dims(np.squeeze(imread(f)),axis=0)
            G_lbl_file = lbl_path+f+'green_labels.tif'
            R_lbl_file = lbl_path+f+'red_labels.tif'

            G_IM_file = im_path+f+'green.tif'
            R_IM_file = im_path+f+'red.tif'
            
            G_lbl = one_hot(get_arr(G_lbl_file))
            R_lbl = one_hot(get_arr(R_lbl_file))
            G_IM = get_arr(G_IM_file)
            R_IM = get_arr(R_IM_file)
            
            IM_list.append(np.expand_dims(np.concatenate([G_IM,R_IM],axis=0),axis=0))
            lbl_list.append(np.expand_dims(np.concatenate([G_lbl,R_lbl],axis=0),axis=0))
        
        #(tiles,channels,x,y)
        self.IM = np.pad(np.concatenate(IM_list,axis=0),pad_width=[[0,0],[0,0],[pad,pad],[pad,pad]],mode='reflect')
        self.lbl = np.pad(np.concatenate(lbl_list,axis=0),pad_width=[[0,0],[0,0],[pad,pad],[pad,pad]],mode='reflect')

        self.n_tiles = self.IM.shape[0]
        self.im_shape = np.array(self.IM.shape)
        self.tile_shape_padded = self.im_shape[-2:]
        self.tile_shape_orig = self.tile_shape_padded-2*pad
        self.patch_size = patch_size
        self.pad = pad
        return

    def __len__(self):
        return self.n_tiles

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_item = self.IM[idx[0],:,idx[1]:idx[1]+self.patch_size,idx[2]:idx[2]+self.patch_size]
        lbl_item = self.lbl[idx[0],:,idx[1]:idx[1]+self.patch_size,idx[2]:idx[2]+self.patch_size]
        return {'im':im_item,'lbl':lbl_item}



class MyRandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        n_tiles (int): number of tiles in the dataset
        max_x (int): max value of x that is used to cut a patch from the tile, i.e. x:x+patch_size
        max_y (int): max value of y that is used to cut a patch from the tile, i.e. y:y+patch_size
        num_samples (int): number of samples to draw. 
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, n_tiles, max_x, max_y, num_samples=None, generator=None):
        self.num_samples = num_samples
        self.n_tiles = n_tiles
        self.max_x = max_x
        self.max_y = max_y
        self.generator = generator

    def __iter__(self):
        tile_rand_tensor = torch.randint(high=self.n_tiles, size=(self.num_samples,), dtype=torch.int64, generator=self.generator)
        imx_rand_tensor = torch.randint(high=self.max_x, size=(self.num_samples,), dtype=torch.int64, generator=self.generator)
        imy_rand_tensor = torch.randint(high=self.max_y, size=(self.num_samples,), dtype=torch.int64, generator=self.generator)
        return iter(zip(tile_rand_tensor.tolist(),imx_rand_tensor.tolist(),imy_rand_tensor.tolist()))

    def __len__(self):
        return self.num_samples