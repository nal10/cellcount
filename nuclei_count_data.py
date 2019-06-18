import numpy as np
import skimage.io as skio
from keras.utils import Sequence, to_categorical


def IOpaths(exp_name='nuclei_count'):
    from pathlib import Path

    curr_path = str(Path().absolute())
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellCount/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellCount/'
    elif '/home/shenqin' in curr_path:
        base_path = '/home/shenqin/Local/CellCount/'
    else: #beaker relative paths
        print('File paths not set!')

    dir_pth={}
    dir_pth['im']          = base_path+'dat/raw/Dataset_03_Images_clean/'
    dir_pth['lbl']         = base_path+'dat/proc/Dataset_03_Labels_v1/'
    dir_pth['result']      = base_path+'dat/results/' + exp_name
    dir_pth['checkpoints'] = base_path+'dat/results/' + exp_name + '/checkpoints/'
    dir_pth['logs']        = base_path+'dat/results/' + exp_name + '/logs/'
    
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True)
    Path(dir_pth['checkpoints']).mkdir(parents=True, exist_ok=True)
    Path(dir_pth['result']).mkdir(parents=True, exist_ok=True)
    return dir_pth


class DataClass(object):
    def __init__(self, paths, file_id, pad):
        im = skio.imread(paths['im'] + file_id + '.tif')/255.0 
        im = np.pad(im, pad, 'reflect')

        lbl = skio.imread(paths['lbl'] + file_id + '_labels.tif')/1.0 #Division by 1. forces float type.
        lbl_zeropadded = np.pad(lbl, pad, 'constant',constant_values=0)
        lbl = np.pad(lbl, pad, 'reflect')

        self.pad = pad
        self.im_lbl = np.stack([im,lbl],axis=-1)
        self.fg_xy = np.nonzero(lbl_zeropadded>0)
        self.fg_xy = np.asarray(self.fg_xy).transpose()

        self._shuffle_fg_xy()
        self.id = file_id
        return

    def _shuffle_fg_xy(self):
        np.random.shuffle(self.fg_xy)
        return

    def __str__(self):
        return "File id : {}".format(self.id)

    def _random_fg_xy(self):
        '''Returns a single random foreground patch midpoint xy. Empty if no fg label.
        '''
        if np.size(self.fg_xy)>0:
            i = np.random.randint(0,np.shape(self.fg_xy)[0])
            xy = self.fg_xy[i]
        else:
            xy = None
        return xy

    def _random_xy(self,n=1):
        '''Returns a single random patch midpoint xy.
        '''
        x = np.random.randint(self.pad,self.im_lbl.shape[0]-self.pad)
        y = np.random.randint(self.pad,self.im_lbl.shape[1]-self.pad)
        
        xy = np.array([[x,y]])
        return xy


class DataGenerator(Sequence):
    '''Generator pops one batch of data at a time
        \n `D_list`: List of class objects from which to generate training data 
    '''
    
    def __init__(self, file_ids, dir_pth, max_fg_frac=0.5, batch_size=4, patch_size=128 ,n_steps_per_epoch=200):
        self.dataObj = [DataClass(paths=dir_pth, file_id=f, pad=int(patch_size)) for f in file_ids]
        for obj in self.dataObj:
            obj._shuffle_fg_xy()
        
        assert patch_size % 2 == 0, 'even patch_size expected'
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_steps_per_epoch = n_steps_per_epoch
        self.max_fg_frac = max_fg_frac
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
            if r<self.max_fg_frac:
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

        return {'input_im':np.expand_dims(self.im,-1)},{'output_im': to_categorical(np.expand_dims(self.lbl,-1),num_classes=3)}
        
    def on_epoch_end(self):
        #Check whether random numbers are repeated every epoch.
        print('\n')
        print(np.random.randint(1000))
        return
