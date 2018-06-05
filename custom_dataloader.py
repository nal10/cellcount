#This code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
#Functions here will
# 1. Perform augmentations on the batch
# 2. Load different subset of data after every training epoch
# This function assumes that mod(samples, batch_size) = 0

# Debug:
# from custom_dataloader import DataGenerator
# import numpy as np
# A = DataGenerator(['53'])
# I.im3dscroll(np.reshape(A.x,(len(A.x), np.size(A.x, 1), np.size(A.x, 2))))

import numpy as np
from keras.utils import Sequence
import fileIO
import random


class DataGenerator(Sequence):
    'This class is a generator for examples used for training keras models'

    def __init__(self, file_ids, batch_size=4, shuffle=True):
        #Load data at initialization the way it is done at the end of each epoch
        'Initialization. The shuffle argument is probably used by keras fit_generator'
        self.batch_size = batch_size
        self.fileids = file_ids
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(np.size(self.x, 0) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indices of the batch
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        'Load new files after each epoch'
        #print('\n Reached epoch end. Loading new files')
        #loadthis = self.fileids  #Loads all files in training dataset simultaneously.
        #loadthis = [random.choice(self.fileids)] #Loads a single randomly selected file for training.
        loadthis = random.sample(self.fileids,4) #Loads 4 single randomly selected file for training.
        isrotate = True
        isflips = True

        #Put the original files into im and labels
        im = fileIO.load_IM(fileid=loadthis)
        labels = fileIO.load_labels(fileid=loadthis)

        #Generate patches. Each list element is 3d patch array constructed from one original file
        im_list = []
        labels_list = []
        for i in range(0, im.shape[0]):
            im_list += [fileIO.gen_patch(im[i], stride=(32, 32))]
            labels_list += [fileIO.gen_patch(
                labels[i], stride=(32, 32))]

        #Stack list elements into a single 3d array
        im_array = fileIO.stack_list(im_list)
        label_array = fileIO.stack_list(labels_list)

        if isrotate:
            #Stack rotated patches and labels
            im_array_rot1 = np.rot90(im_array, 1, (1, 2))
            im_array_rot2 = np.rot90(im_array, 2, (1, 2))
            im_array_rot3 = np.rot90(im_array, 3, (1, 2))

            im_array = np.append(im_array, im_array_rot1, axis=0)
            im_array = np.append(im_array, im_array_rot2, axis=0)
            im_array = np.append(im_array, im_array_rot3, axis=0)

            del im_array_rot1, im_array_rot2, im_array_rot3

            label_array_rot1 = np.rot90(label_array,1,(1,2))
            label_array_rot2 = np.rot90(label_array,2,(1,2))
            label_array_rot3 = np.rot90(label_array,3,(1,2))

            label_array = np.append(label_array, label_array_rot1, axis=0)
            label_array = np.append(label_array, label_array_rot2, axis=0)
            label_array = np.append(label_array, label_array_rot3, axis=0)

            del label_array_rot1, label_array_rot2, label_array_rot3

        if isflips:
            im_array_flip1 = np.flip(im_array, axis=1)
            im_array_flip2 = np.flip(im_array, axis=2)

            im_array = np.append(im_array, im_array_flip1, axis=0)
            im_array = np.append(im_array, im_array_flip2, axis=0)

            del im_array_flip1, im_array_flip2

            label_array_flip1 = np.flip(label_array, axis=1)
            label_array_flip2 = np.flip(label_array, axis=2)

            label_array = np.append(label_array, label_array_flip1, axis=0)
            label_array = np.append(label_array, label_array_flip2, axis=0)
            
            del label_array_flip1, label_array_flip2
    
        #Reshape to add the 4th dimension
        im_array = np.reshape(im_array, (len(im_array), np.size(
            im_array, 1), np.size(im_array, 2), 1))
        label_array = np.reshape(label_array, (len(label_array), np.size(
            label_array, 1), np.size(label_array, 2), 1))

        self.x = im_array.astype('float32') / 255.
        self.y = label_array.astype('float32') / 1.

        return {'input_im': self.x, 'output_im': self.y}

class DataGeneratorVal(Sequence):
    'This class is a generator for examples used for training keras models'

    def __init__(self, file_ids, batch_size=4, shuffle=True):
        #Load data at initialization the way it is done at the end of each epoch
        'Initialization. The shuffle argument is probably used by keras fit_generator'
        self.batch_size = batch_size
        self.fileids = file_ids
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(np.size(self.x, 0) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indices of the batch
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        'Load new files after each epoch'
        #print('\n Reached epoch end. Loading new files')
        loadthis = self.fileids  #Loads all files in validation set simultaneously.
        #loadthis = [random.choice(self.fileids)] #Loads a single randomly selected file for training.
        #loadthis = random.sample(self.fileids,1) #Loads 4 single randomly selected file for training.
        isrotate = False
        isflips = False

        #Put the original files into im and labels
        im = fileIO.load_IM(fileid=loadthis)
        labels = fileIO.load_labels(fileid=loadthis)

        #Generate patches. Each list element is 3d patch array constructed from one original file
        im_list = []
        labels_list = []
        for i in range(0, im.shape[0]):
            im_list += [fileIO.gen_patch(im[i], stride=(64, 64))]
            labels_list += [fileIO.gen_patch(
                labels[i], stride=(64, 64))]

        #Stack list elements into a single 3d array
        im_array = fileIO.stack_list(im_list)
        label_array = fileIO.stack_list(labels_list)

        if isrotate:
            #Stack rotated patches and labels
            im_array_rot1 = np.rot90(im_array, 1, (1, 2))
            im_array_rot2 = np.rot90(im_array, 2, (1, 2))
            im_array_rot3 = np.rot90(im_array, 3, (1, 2))

            im_array = np.append(im_array, im_array_rot1, axis=0)
            im_array = np.append(im_array, im_array_rot2, axis=0)
            im_array = np.append(im_array, im_array_rot3, axis=0)

            del im_array_rot1, im_array_rot2, im_array_rot3

            label_array_rot1 = np.rot90(label_array,1,(1,2))
            label_array_rot2 = np.rot90(label_array,2,(1,2))
            label_array_rot3 = np.rot90(label_array,3,(1,2))

            label_array = np.append(label_array, label_array_rot1, axis=0)
            label_array = np.append(label_array, label_array_rot2, axis=0)
            label_array = np.append(label_array, label_array_rot3, axis=0)

            del label_array_rot1, label_array_rot2, label_array_rot3

        if isflips:
            im_array_flip1 = np.flip(im_array, axis=1)
            im_array_flip2 = np.flip(im_array, axis=2)

            im_array = np.append(im_array, im_array_flip1, axis=0)
            im_array = np.append(im_array, im_array_flip2, axis=0)

            del im_array_flip1, im_array_flip2

            label_array_flip1 = np.flip(label_array, axis=1)
            label_array_flip2 = np.flip(label_array, axis=2)

            label_array = np.append(label_array, label_array_flip1, axis=0)
            label_array = np.append(label_array, label_array_flip2, axis=0)
            
            del label_array_flip1, label_array_flip2
    
        #Reshape to add the 4th dimension
        im_array = np.reshape(im_array, (len(im_array), np.size(
            im_array, 1), np.size(im_array, 2), 1))
        label_array = np.reshape(label_array, (len(label_array), np.size(
            label_array, 1), np.size(label_array, 2), 1))

        self.x = im_array.astype('float32') / 255.
        self.y = label_array.astype('float32') / 1.

        return {'input_im': self.x, 'output_im': self.y}