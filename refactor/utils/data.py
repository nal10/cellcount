import glob
from aicsimageio import imread
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from pathlib import Path

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
                 lbl_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Unet_tiles_082020/',
                 subset = 'train',
                 one_hot_labels = False,
                 np_transform=None,
                 torch_transforms=None):
        

        super().__init__()
        file_list = self.get_file_list(subset=subset,im_path=im_path)

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
            
            if one_hot_labels:
                G_lbl = one_hot(get_arr(G_lbl_file))
                R_lbl = one_hot(get_arr(R_lbl_file))
            else:
                G_lbl = get_arr(G_lbl_file)
                R_lbl = get_arr(R_lbl_file)

            G_IM = get_arr(G_IM_file)
            R_IM = get_arr(R_IM_file)

            assert G_IM.dtype=='uint8', "transform pipeline tested only for uint8 input"
            assert R_IM.dtype=='uint8', "transform pipeline tested only for uint8 input"
            
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
        self.np_transform = np_transform
        self.torch_transforms = torch_transforms
        return

    def __len__(self):
        return self.n_tiles

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_item = self.IM[idx[0],:,idx[1]:idx[1]+self.patch_size,idx[2]:idx[2]+self.patch_size]
        lbl_item = self.lbl[idx[0],:,idx[1]:idx[1]+self.patch_size,idx[2]:idx[2]+self.patch_size]

        #Flipping arrays is much more efficient if directly performed on numpy arrays first
        if self.np_transform is not None:
            im_item,lbl_item = self.np_transform(im_item,lbl_item)

        im_item = torch.as_tensor(im_item)
        lbl_item = torch.as_tensor(lbl_item)

        if self.torch_transforms is not None:
            return self.torch_transforms({'im':im_item,'lbl':lbl_item})
        return {'im':im_item,'lbl':lbl_item}

    def get_file_list(self,subset,im_path):
        """ 
        Args:
            subset (str): 'train','val' or 'all'
            im_path (str): 

        Returns:
            list of filenames
        """
        validation_file_list = ['527100_1027993339_0065_tile_9_8_', '527103_1027700130_0060_tile_7_3_','529690_1030079321_0085_tile_8_13_']
        file_list = glob.glob(im_path+'/*_green.tif')
        file_list = glob.glob(im_path+'*_green.tif')
        file_list = [f.split('green.tif')[0] for f in file_list]
        file_list = [f.split(im_path)[1] for f in file_list]
        assert len(file_list)>0, "Files not found at {}".format(im_path)

        if subset=='train':
            x = list(set(file_list)-set(validation_file_list))
            print("Found {} files".format(len(x)))
            return x 
        elif subset=='val':
            x = set(file_list)
            x = list(x.intersection(set(validation_file_list)))
            print("Found {} files".format(len(x)))
            return x
        elif subset=='all':
            x = list(set(file_list))
            print("Found {} files".format(len(x)))
            return x
        else:
            print('name a valid subset')


class MyRandomSampler(Sampler):
    r"""Returns indices to randomly specify patches tiles in the dataset.

    Arguments:
        n_tiles (int): number of tiles in the dataset
        max_x (int): max value of x that is used to cut a patch from the tile, i.e. x:x+patch_size
        max_y (int): max value of y that is used to cut a patch from the tile, i.e. y:y+patch_size
        num_samples (int): number of samples to draw. 
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, n_tiles, min_x=0, min_y=0, max_x=2048, max_y=2048, num_samples=None, generator=None):
        self.num_samples = num_samples
        self.n_tiles = n_tiles
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.generator = generator

    def __iter__(self):
        tile_rand_tensor = torch.randint(high=self.n_tiles,
                                         size=(self.num_samples,), dtype=torch.int64, generator=self.generator)
        imx_rand_tensor = torch.randint(low=self.min_x, high=self.max_x,
                                        size=(self.num_samples,), dtype=torch.int64, generator=self.generator)
        imy_rand_tensor = torch.randint(low=self.min_y, high=self.max_y,
                                        size=(self.num_samples,), dtype=torch.int64, generator=self.generator)
        return iter(zip(tile_rand_tensor.tolist(), imx_rand_tensor.tolist(), imy_rand_tensor.tolist()))

    def __len__(self):
        return self.num_samples


#==============================================================================
class Pred_ai224_RG(Dataset):
    """Dataset for training images

    Args:
        im_path: directory path with image .tif files 
        patch_size: patch_size of returned items
        output_size: output size of the network.
    """

    def __init__(self,
                 patch_size = 260,
                 output_size = 172,
                 im_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Unet_tiles_082020/',
                 fname='527100_1027993339_0065_tile_9_8_'):
        
        super().__init__()
        G_IM_file=im_path+fname+'green.tif'
        R_IM_file=im_path+fname+'red.tif'
        assert Path(G_IM_file).is_file(), f'{G_IM_file} not found'
        assert Path(R_IM_file).is_file(), f'{R_IM_file} not found'
        get_arr = lambda f: np.expand_dims(np.squeeze(imread(f)),axis=0)
        
        IM_list = []
        G_IM = get_arr(G_IM_file)
        R_IM = get_arr(R_IM_file)

        assert G_IM.dtype=='uint8', "transform pipeline tested only for uint8 input"
        assert R_IM.dtype=='uint8', "transform pipeline tested only for uint8 input"
        
        #(tiles,channels,x,y)
        IM_list.append(np.expand_dims(np.concatenate([G_IM,R_IM],axis=0),axis=0))

        IM_shape = IM_list[0].shape[-2:]
        
        n_x_patches = np.ceil(IM_shape[0]/output_size).astype(int)
        n_y_patches = np.ceil(IM_shape[1]/output_size).astype(int)

        pad_xi = int((patch_size-output_size)/2)
        pad_yi = int((patch_size-output_size)/2)
        pad_xf = int(n_x_patches*output_size + (patch_size-output_size)/2 - IM_shape[0])
        pad_yf = int(n_y_patches*output_size + (patch_size-output_size)/2 - IM_shape[1])
        
        self.IM = np.pad(np.concatenate(IM_list,axis=0),pad_width=[[0,0],[0,0],[pad_xi,pad_xf],[pad_yi,pad_yf]],mode='reflect')
        
        self.n_tiles = self.IM.shape[0]
        self.tile_shape_orig = IM_shape[-2:]
        self.tile_shape_padded = np.array(self.IM.shape[-2:])
        self.patch_size = patch_size
        self.output_size = output_size

        self.pad_xi = pad_xi
        self.pad_yi = pad_yi
        self.pad_xf = pad_xf
        self.pad_yf = pad_yf

        self.n_x_patches = n_x_patches
        self.n_y_patches = n_y_patches
        return

    def __len__(self):
        return self.n_tiles

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_item = self.IM[idx[0],:,idx[1]:idx[1]+self.patch_size,idx[2]:idx[2]+self.patch_size]
        im_item = torch.as_tensor(im_item)
        idx = torch.as_tensor(idx)
        return {'im':im_item, 'idx':idx[-2:]}

class Pred_Sampler(Sampler):
    r"""Returns indices to sequentially generate patches as per dataset spec.
    Use.

    Arguments:
        dataset: 
    """

    def __init__(self, dataset):
        self.n_tiles = int(dataset.n_tiles)
        self.output_size= int(dataset.output_size)
        self.n_x_patches = int(dataset.n_x_patches)
        self.n_y_patches = int(dataset.n_y_patches)
        
    def __iter__(self):
        for t in range(self.n_tiles):
            for i in range(self.n_x_patches):
                for j in range(self.n_y_patches):
                    #print(t, int(i*self.output_size),int(j*self.output_size))
                    yield([t, i*self.output_size, j*self.output_size])

    def __len__(self):
        return self.n_x_patches*self.n_y_patches