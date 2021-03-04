import glob
import zarr
import dask.array as da
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from aicsimageio import imread
from pathlib import Path
from timebudget import timebudget as time


def convert_u16_to_u8(im_u16):
    """Converts a uint16 image into a uint8 image after scaling. 
    This function matches scripts used to produce uint8 .tif files from LIMS jp2 images. 
    """
    assert im_u16.dtype == 'uint16', "Input must be uint16"
    return np.uint8(np.round(im_u16.astype(float)*(2**8)/(2**16-1)))


def jp2_to_zarr(file_list, destination:str,overwrite=False):
    """
    """
    #NOT TESTED
    import glymur
    glymur.set_option('lib.num_threads', 2)

    for file in file_list:
        if file.endswith(".jp2") and overwrite:
            zarr_file = file.split('.')[0] + '.zarr'
            jp2 = glymur.Jp2k(file)
            zarr.save(destination+zarr_file, jp2[:])
            print(f'Wrote {destination+zarr_file}')
        elif overwrite:
            print(f'====> Ignored: {file}')
    return



def nonzero_row_col(zarr_file,verbose=True):
    """Output the co-ordinates where the first non-zero rows and columns occur in a given zarr array. 
    Column- and row-wise sum calculations take ~50s for a 30000 x 40000 x 3 zarr image.

    Example:
    ```
    im_path = '~/Remote-AI-root/allen/programs/celltypes/workgroups/mct-t200/Molecular_Genetics_Daigle_Team/Elyse/Unet_WB_testing/547656/'
    fname_list = glob.glob(im_path+'/*.zarr')
    for fname in fname_list:
        f = zarr.open(fname)
        print(fname)
        _ = nonzero_row_col(f)
    ```
    """

    dx = da.from_array(zarr_file)

    #Reduce channel dimension
    dflat = dx.sum(axis=2)

    #Calculate column non-zeros
    colsum = dflat.sum(axis=0)
    colsum = colsum.compute()
    y = np.diff(colsum > 0)
    miny = np.min(np.flatnonzero(y != 0))
    maxy = np.max(np.flatnonzero(y != 0))

    #Calculate row non-zeros
    rowsum = dflat.sum(axis=1)
    rowsum = rowsum.compute()
    x = np.diff(rowsum > 0)
    minx = np.min(np.flatnonzero(x != 0))
    maxx = np.max(np.flatnonzero(x != 0))

    if verbose:
        print(f'x-limits: [{minx}, {maxx}],y-limits: [{miny}, {maxy}]. Image shape: {zarr_file.shape}]')
    return minx, maxx, miny, maxy


class Ai224_RG_Dataset(Dataset):
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


class RandomSampler(Sampler):
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


class Pred_Ai224_RG_Dataset(Dataset):
    """Dataset with only images, used for prediction with the model

    Args:
        patch_size: patch_size of returned items
        output_size: output size of the network
        im_path: directory path with image .tif files 
        fname: name of image without channel extension (see default argument)
        scale: pixel values are divided by this number
    """

    def __init__(self,
                 patch_size = 260,
                 output_size = 172,
                 im_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Unet_tiles_082020/',
                 fname='527100_1027993339_0065_tile_9_8_',
                 scale=1.0):
        
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
        self.scale=float(scale)

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
        im_item = self.IM[idx[0],:,idx[1]:idx[1]+self.patch_size,idx[2]:idx[2]+self.patch_size]/self.scale
        im_item = torch.as_tensor(im_item)
        idx = torch.as_tensor(idx)
        return {'im':im_item, 'idx':idx[-2:]}


class Pred_Sampler(Sampler):
    r"""Returns indices to sequentially generate patches as per dataset spec.

    Arguments:
        dataset: object of a Dataset class
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


class Pred_Ai224_RG_Zarr(Dataset):
    """Read zarr dataset for prediction with the model

    Args:
        patch_size: patch_size of returned items
        output_size: output size of the network
        im_path: directory path with image .zarr files 
        fname: name of image without channel extension (see default argument)
        scale: pixel values are divided by this number
    """

    def __init__(self,
                 patch_size = 260,
                 output_size = 172,
                 im_path='/home/rohan/Remote-AI-root/allen/programs/celltypes/workgroups/mct-t200/Molecular_Genetics_Daigle_Team/Elyse/Unet_WB_testing/547656/',
                 fname=None,
                 scale=1.0):
        
        super().__init__()
        self.IM = zarr.open(im_path+fname)
        #[15000:20000,10000:15000,:]
        self.IM = self.IM
        IM_shape = self.IM.shape[:2]
        
        n_x_patches = np.ceil(IM_shape[0]/output_size).astype(int)
        n_y_patches = np.ceil(IM_shape[1]/output_size).astype(int)

        #Not implementing padding for zarr files. 
        #The images in this dataset already have enough padding around the tissue.
        pad_xi = int((patch_size-output_size)/2)
        pad_yi = int((patch_size-output_size)/2)
        pad_xf = int(n_x_patches*output_size + (patch_size-output_size)/2 - IM_shape[0])
        pad_yf = int(n_y_patches*output_size + (patch_size-output_size)/2 - IM_shape[1])
        
        self.n_tiles = self.IM.shape[0]
        self.tile_shape_orig = IM_shape
        self.tile_shape_padded = IM_shape
        self.patch_size = patch_size
        self.output_size = output_size
        self.scale=float(scale)

        self.pad_xi = pad_xi
        self.pad_yi = pad_yi
        self.pad_xf = pad_xf
        self.pad_yf = pad_yf

        self.n_x_patches = n_x_patches
        self.n_y_patches = n_y_patches
        self.dummy = np.zeros((2,self.patch_size,self.patch_size))
        return

    def __len__(self):
        return self.n_tiles

    def __getitem__(self, idx):
        #idx is tuple with (t,x,y) co-ordinates defining the tile index, and current patch corner.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t,x,y = idx

        #If out of bounds:
        if (x+self.patch_size >= self.tile_shape_orig[0]) or (y+self.patch_size >= self.tile_shape_orig[1]):
            return {'im':torch.as_tensor(self.dummy), 'idx':torch.as_tensor([x,y])}

        #Ignore idx[0]: it specifies a tile. zarr based dataloader is expected to work with a single image section.
        im_item = self.IM[x:x+self.patch_size, y:y+self.patch_size, :2]

        #zarr files have uint16 data. only training and augmentation requires uint8.
        im_item = im_item.astype(float)*(2**8)/(2**16-1)/self.scale
        
        #zarr file shape (x,y,[red,green]), with 1:green and 0:red. Model requires ([green,red],x,y)
        #copy is required because the numpy array view has negative steps
        im_item = np.moveaxis(np.flip(im_item,axis=2),source=2,destination=0).copy()
        
        return {'im':torch.as_tensor(im_item), 'idx':torch.as_tensor([x,y])}


class Pred_Sampler_Zarr(Sampler):
    r"""Returns indices to sequentially generate patches as per zarr spec.
     - tile -> subtile -> patch
     - subtiles are defined so that prediction image -> xy co-ordinates encounters fewer patch boundaries.
    
    Arguments:
        dataset: object of a Dataset class
    """

    def __init__(self, dataset, n_x_patch_per_subtile, n_y_patch_per_subtile):
        self.n_tiles = int(dataset.n_tiles)
        self.output_size = int(dataset.output_size)
        self.n_x_patches = int(dataset.n_x_patches)
        self.n_y_patches = int(dataset.n_y_patches)
        self.n_x_patch_per_subtile = n_x_patch_per_subtile
        self.n_y_patch_per_subtile = n_y_patch_per_subtile
        self.n_x_subtiles = np.ceil(self.n_x_patches/self.n_x_patch_per_subtile).astype(int)
        self.n_y_subtiles = np.ceil(self.n_y_patches/self.n_y_patch_per_subtile).astype(int)
        
        
    def __iter__(self):
        for t in range(self.n_tiles):
            for i_x_subtile in range(self.n_x_subtiles):
                for i_y_subtile in range(self.n_y_subtiles):
                    for i in range(self.n_x_patch_per_subtile):
                        for j in range(self.n_y_patch_per_subtile):
                            yield [t,
                                   (self.n_x_patch_per_subtile*i_x_subtile + i)*self.output_size,
                                   (self.n_y_patch_per_subtile*i_y_subtile + j)*self.output_size]

    def __len__(self):
        return self.n_x_patches*self.n_y_patches


def replace_duplicates(points, distance_thr):
    """Points that are within threshold distance are considered as duplicates. 
    The same cell can be partially segmented and detected > 1 sections by the U-net based method. 
    Duplicates replaced with the centroid of duplicates. 

    Args:
        points: numpy array of shape (n x 3) or (n x 2)

    Returns:
        points: after duplicates are replaced by the corresponding centroid
    """

    return points
    # import matplotlib.pyplot as plt
    # >>> import numpy as np
    # >>> 
    # >>> np.random.seed(21701)
    # >>> points = np.random.random((20, 2))
    # >>> plt.figure(figsize=(6, 6))
    # >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)
    # >>> kd_tree = cKDTree(points)
    # >>> pairs = kd_tree.query_pairs(r=0.2)
    # >>> for (i, j) in pairs:
    # ...     plt.plot([points[i, 0], points[j, 0]],
    # ...             [points[i, 1], points[j, 1]], "-r")
    # >>> plt.show()