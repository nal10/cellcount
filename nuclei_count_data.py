import numpy as np
import skimage.io as skio
from tensorflow.keras.utils import to_categorical,Sequence
import pdb


def IOpaths(exp_name='nuclei_count'):
    from pathlib import Path

    curr_path = str(Path().absolute())
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellCount/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellCount/'
    else:
        print('File paths not set!')

    dir_pth={}
    dir_pth['im']          = base_path+'dat/raw/Dataset_03_Images_clean/'
    dir_pth['lbl']         = base_path+'dat/proc/Dataset_03_Labels_v1/'
    dir_pth['result']      = base_path+'dat/results/' + exp_name + '/'
    dir_pth['checkpoints'] = base_path+'dat/results/' + exp_name + '/checkpoints/'
    dir_pth['logs']        = base_path+'dat/results/' + exp_name + '/logs/'
    
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True)
    Path(dir_pth['checkpoints']).mkdir(parents=True, exist_ok=True)
    Path(dir_pth['result']).mkdir(parents=True, exist_ok=True)
    return dir_pth


class DataClass(object):
    BACKGROUND=0
    BOUNDARY=1
    FOREGROUND=2
    
    def __init__(self, paths, file_id, pad):
        """Creates an object with numpy arrays consisting of images+labels+foreground xy positions for a single training dataset file. 
        
        Arguments:
            paths: dictionary with path locations stored in keys 'lbl' and 'im'.
            file_id: string that identifies a specific image.
            pad: size of the padding.
        """        
        padding_mode = 'reflect' #To reduce potential edge effects
        im = skio.imread(paths['im'] + file_id + '.tif')/255.0 
        im = np.pad(array=im, pad_width=pad, mode=padding_mode)

        lbl = skio.imread(paths['lbl'] + file_id + '_labels.tif')/1.0 #Division by 1. forces float type.
        lbl_zeropadded = np.pad(lbl, pad, 'constant',constant_values=0) #Used to determine true foreground positions, while excluding the padded regions
        lbl = np.pad(array=lbl, pad_width=pad, mode=padding_mode) 

        self.id = file_id
        self.pad = pad
        self.im_lbl = np.stack([im,lbl],axis=-1) #has dimensions of x_size x y_size x 2
        self.fg_xy = np.transpose(np.asarray(np.nonzero(lbl_zeropadded!=self.BACKGROUND)))
        if self.fg_xy.size>0:
            self.fg_xy = np.reshape(self.fg_xy,newshape=(int(self.fg_xy.size/2),2))
        else:
            self.fg_xy = np.empty(shape=(0,2),dtype=int)
        return

    def __str__(self):
        return "File id : {}".format(self.id)

    def _shuffle_fg_xy(self):
        '''
        Shuffles fg_xy order. Done to ensure batches are not identical.
        np.random.shuffles(): performs in-place shuffle along the first axis. Empties are handled as expected.
        s'''
        np.random.shuffle(self.fg_xy) 
        return

    def _random_fg_xy(self):
        '''Returns 
        a single random foreground patch midpoint xy. Empty if no fg label.
        '''
        if np.size(self.fg_xy)>0:
            i = np.random.randint(0,np.shape(self.fg_xy)[0])
            xy = self.fg_xy[i,:]
        else:
            xy = None
        return xy

    def _random_xy(self):
        '''Returns a single random patch midpoint xy.
        '''
        xy = np.zeros(shape=(2,),dtype=int)
        xy[0] = np.random.randint(self.pad,self.im_lbl.shape[0]-self.pad,size=1)
        xy[1] = np.random.randint(self.pad,self.im_lbl.shape[1]-self.pad,size=1)
        return xy


def random_trainbatch(dataObj_list=[], min_fg_frac=0.5, patch_size=128, batch_size=10):
    '''
    Samples files uniformly. 
    `min_fg_frac` only applies to files with atleast 1 fg pixel
    '''
    pdb.set_trace()
    fid = np.random.randint(low=0, high=len(
        dataObj_list), size=(batch_size,), dtype=int)
    im = np.zeros((batch_size, patch_size, patch_size))
    lbl = np.zeros((batch_size, patch_size, patch_size))
    batch_ind = 0
    for f in fid:
        xy = None
        r = np.random.rand()
        if r < min_fg_frac:  # Choose one foreground pixel (may be empty)
            xy = dataObj_list[f]._random_fg_xy()
        if xy is None:
            xy = dataObj_list[f]._random_xy()
        xy = np.squeeze(xy)
        shift_xy = np.random.randint(low=-dataObj_list[f].pad,
                                     high=dataObj_list[f].pad,
                                     size=xy.shape)
        xy = xy+shift_xy
        half_patch = int(patch_size/2)
        pdb.set_trace()
        im_lbl = dataObj_list[f].im_lbl[xy[0]-half_patch:xy[0]+half_patch,
                                        xy[1]-half_patch:xy[1]+half_patch,
                                        :]
        im[batch_ind, :, :] = im_lbl[:, :, 0]
        lbl[batch_ind, :, :] = im_lbl[:, :, 1]
        batch_ind = batch_ind+1

    im = np.expand_dims(im, -1)
    cat_lbl = to_categorical(np.expand_dims(lbl, -1), num_classes=3)
    return ({'input_im': im}, {'output_im': cat_lbl})
    

def trainingData(dataObj_list, min_fg_frac=0.8, patch_size=128, n_patches_perfile=20):
    """Calculate image and categorical label patches using DataClass objects. 
    Batches necessarily have patches from each file, evenly sampled from the file.  
    
    Arguments:
        dataObj_list: a list of DataClass objects 
    
    Keyword Arguments:
        min_fg_frac {float} -- Minimum fraction of patches per batch that have at least one foreground pixels (default: {0.8})
        patch_size {int} -- This is matched to the network input.
        n_patches_perfile {int} -- number of patches per file
    
    Returns:
        Tuple of input and output dictionaries. 
        The input and output dicts contsist of 4d arrays (x,y,n_patches,channels)
    """    
    total_patch_count=0
    im =  np.zeros((len(dataObj_list)*n_patches_perfile,patch_size,patch_size))
    lbl = np.zeros((len(dataObj_list)*n_patches_perfile,patch_size,patch_size))
    for d in range(len(dataObj_list)):
        count_thisfile=0
        while count_thisfile<n_patches_perfile: #Keep choosing central pixels to make patches around
            r = np.random.rand()
            if r<min_fg_frac:
                #Choose a single foreground pixel (may be empty)
                ind = dataObj_list[d]._random_fg_xy() 
            else:
                #Choose a single random pixel
                ind = dataObj_list[d]._random_xy()
                
            if ind is not None:
                ind = np.squeeze(ind)
                im_lbl = dataObj_list[d].im_lbl[ind[0]-int(patch_size/2):ind[0]+int(patch_size/2),
                                                ind[1]-int(patch_size/2):ind[1]+int(patch_size/2),:]
                
                im[total_patch_count,:,:] = im_lbl[:,:,0]
                lbl[total_patch_count,:,:] = im_lbl[:,:,1]
                count_thisfile = count_thisfile + 1
                total_patch_count = total_patch_count + 1
    im = np.expand_dims(im, -1)
    cat_lbl = to_categorical(np.expand_dims(lbl, -1),num_classes=3)
    return ({'input_im': im}, {'output_im': cat_lbl})

class DataGenerator(Sequence):
    '''Generator pops one batch of data at a time. Shift, rotation and flip augmentations are performed at random.
        \n `D_list`: List of class objects from which to generate training data 
    '''
    
    def __init__(self, file_ids, dir_pth, min_fg_frac=0.5, batch_size=4, patch_size=128 ,n_steps_per_epoch=200):
        self.dataObj = [DataClass(paths=dir_pth, file_id=f, pad=int(patch_size)) for f in file_ids]
        for obj in self.dataObj:
            obj._shuffle_fg_xy()
        
        assert patch_size % 2 == 0, 'even patch_size expected'
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_steps_per_epoch = n_steps_per_epoch
        self.min_fg_frac = min_fg_frac
        self.max_shift = np.floor(patch_size/2).astype(int)
        
        self.im = np.zeros((batch_size,patch_size,patch_size))
        self.lbl = np.zeros((batch_size,patch_size,patch_size))
        return
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_steps_per_epoch
        
    def __getitem__(self, idx):
        'Generate one batch of data'
        count=0
        while count<self.batch_size:
            r = np.random.rand()
            d = np.random.randint(len(self.dataObj))
            #print(r,d)
            if r<self.min_fg_frac:
                ind = self.dataObj[d]._random_fg_xy()
                #Shift foreground images so that it's not always the central pixel
                if ind is not None:
                    ind = ind + np.random.randint(-self.max_shift,self.max_shift,size=np.shape(ind))
            else:
                ind = self.dataObj[d]._random_xy()
                
            if ind is not None:
                ind = np.squeeze(ind)
                im_lbl = self.dataObj[d].im_lbl[ind[0]-int(self.patch_size/2):ind[0]+int(self.patch_size/2),
                                                ind[1]-int(self.patch_size/2):ind[1]+int(self.patch_size/2),:]
                
                #Rotation and flip augmentations
                aug = np.random.randint(5)
                if aug==0:
                    pass
                elif aug==1:
                    im_lbl = np.rot90(im_lbl, k=1, axes=(0, 1))
                elif aug==2:
                    im_lbl = np.rot90(im_lbl, k=3, axes=(0, 1))
                elif aug==3:
                    im_lbl = np.flip(im_lbl, axis=1)
                elif aug==4:
                    im_lbl = np.flip(im_lbl, axis=0)
                
                self.im[count,:,:] = im_lbl[:,:,0]
                self.lbl[count,:,:] = im_lbl[:,:,1]
                count = count+1
        
        im = np.expand_dims(self.im,-1)
        cat_lbl = to_categorical(np.expand_dims(self.lbl,-1),num_classes=3)
        return ({'input_im': im}, {'output_im': cat_lbl})
        
    def on_epoch_end(self):
        return


def validationData(file_ids, dir_pth, min_fg_frac=0.8, patch_size=128, n_patches_perfile=20):
    '''Returns fixed set of patches from validation file list'''

    dataObj_list = [DataClass(paths=dir_pth, file_id=f, pad=int(patch_size)) for f in file_ids]
    
    all_count=0
    im = np.zeros((len(dataObj_list)*n_patches_perfile,patch_size,patch_size))
    lbl = np.zeros((len(dataObj_list)*n_patches_perfile,patch_size,patch_size))
    for d in range(len(dataObj_list)):
        count_thisfile=0
        while count_thisfile<n_patches_perfile:
            r = np.random.rand()
            if r<min_fg_frac:
                ind = dataObj_list[d]._random_fg_xy()
            else:
                ind = dataObj_list[d]._random_xy()
                
            if ind is not None:
                ind = np.squeeze(ind)
                im_lbl = dataObj_list[d].im_lbl[ind[0]-int(patch_size/2):ind[0]+int(patch_size/2),
                                                ind[1]-int(patch_size/2):ind[1]+int(patch_size/2),:]
                
                im[all_count,:,:] = im_lbl[:,:,0]
                lbl[all_count,:,:] = im_lbl[:,:,1]
                count_thisfile = count_thisfile + 1
                all_count = all_count + 1
    return {'input_im':np.expand_dims(im,-1)},{'output_im': to_categorical(np.expand_dims(lbl,-1),num_classes=3)}


