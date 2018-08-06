import random
import numpy as np
from keras.utils import Sequence

class DataGenerator(Sequence):
    '''Create new batch of randomly chosen patches for each epoch. 
        \n Inputs:
        \n file_id: List of strings specifying file id. e.g. ['53','68'] 
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = self.dataset.batch_size
        self.on_epoch_end()
        return
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(np.size(self.im_epoch, 0) / self.batch_size))
        
    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indices of the batch
        im_batch = self.im_epoch[idx * self.batch_size:(idx + 1) * self.batch_size]
        lbl_batch = self.lbl_epoch[idx * self.batch_size:(idx + 1) * self.batch_size]
        return im_batch, {'output_im_1': np.array(lbl_batch == 1).astype('float'),
                          'output_im_2': np.array(lbl_batch == 2).astype('float')}
        
    def on_epoch_end(self):
        self.im_epoch,self.lbl_epoch = self.dataset.get_patches()
        return
