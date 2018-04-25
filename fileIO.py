"""Contains functions to load images, labels, weights and patches."""

# Following shows requirements for the Unet
# def load_data():
# 		# Replace with fileIO functions
# 		im_train = []
# 		im_train_labels = []
# 		im_test = []
# 		im_test_labels = []
# 		return im_train, im_train_labels, im_test, im_test_labels

import sys
import socket
import numpy as np
import skimage.io
import pdb

def set_paths():
    """Return the pathname of the data directory."""
    hostname = socket.gethostname()
    if hostname == 'Fruity.local':
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellCount/'
        rel_IM_path = 'dat/raw/cellSegmentationDataset_v2/'
        rel_label_path = 'dat/raw/cellSegmentationDataset_v2/'
    print('Base path set to: ' + base_path)
    return base_path, rel_IM_path, rel_label_path


def get_fileid():
    """Return fileids for _raw.tif images in directory."""
    import os
    import glob

    base_path, rel_IM_path, _ = set_paths()
    search_pattern = base_path + rel_IM_path + '*raw.tif'
    print('Searching for pattern: ' + search_pattern)
    IMnames = glob.glob(search_pattern)

    #List id of filenames ending with _raw.tif
    fileidlist = []
    for f in IMnames:
        os.path.basename(f)
        os.path.dirname(f)
        os.path.splitext(f)
        temp = os.path.splitext(os.path.basename(f))
        temp = temp[0].split('_')
        fileidlist.append(temp[0])
    return fileidlist


def load_IM(**kwargs):
    """Return images in numpy array for all files listed in fileid."""
    base_path, rel_IM_path, _ = set_paths()
    if 'fileid' in kwargs:
        fileid = kwargs.pop('fileid')
    else:
        fileid = get_fileid()

    # Hardcoded values below for dimensions and number of channels!
    IM = np.zeros((len(fileid), 2500, 2500))
    for idx, item in enumerate(fileid):
        fname = base_path + rel_IM_path + item + '_raw.tif'
        thisIM = skimage.io.imread(fname)
        if len(thisIM.shape)==3:
            IM[idx] = np.copy(thisIM[:, :, 0])
        else:
            IM[idx] = np.copy(thisIM[:, :])
    return IM


def load_labels(**kwargs):
    """Return labels in numpy array for all files listed in fileid.
    _newlabel.tif files were generated with Matlab code - the original 
    label files could not be opened with scikit-image functions

    Keyword arguments:
    fileid -- string containing number identifying an image file
    """

    base_path, _, rel_label_path = set_paths()
    if 'fileid' in kwargs:
        fileid = kwargs.pop('fileid')
    else:
        fileid = get_fileid()

    # Hardcoded values below for dimensions and number of channels!
    labels = np.zeros((len(fileid), 2500, 2500))
    for idx, item in enumerate(fileid):
        fname = base_path + rel_label_path + item + '_newlabels.tif' 
        thislabel = skimage.io.imread(fname)
        if len(thislabel.shape)==3:
            labels[idx] = np.copy(thislabel[:, :, 0])
        else:
            labels[idx] = np.copy(thislabel[:, :])
    return labels


def calc_W(IM,label):
    """Return weights in numpy array for all given Image and label.
    IM and label should have same dimensions. Transformations to obtain
    W are performed in 2D"""

    W = np.zeros(IM.shape)
    return W


def gen_patch(X, **kwargs):
    """Return inputs shaped to 512 x 512 as obtained by strides.
    stride is expected as a 2 element tuple
    """
    if 'stride' in kwargs:
        stride = kwargs.pop('stride')
    else:
        stride = (256, 256)
    sizeIM = (512, 512)
    
    maxstrides = np.zeros(2)
    maxstrides[0] = np.ceil((X.shape[0]-sizeIM[0])/stride[0])+1
    maxstrides[1] = np.ceil((X.shape[1]-sizeIM[1])/stride[1])+1

    start_x = np.arange(0, maxstrides[0], 1) * stride[0]
    start_x = np.append(start_x, X.shape[0] - sizeIM[0])
    start_x = start_x[start_x <= (X.shape[0] - sizeIM[0])]
    start_x = np.unique(start_x)

    start_y = np.arange(0, maxstrides[1], 1) * stride[1]
    start_y = np.append(start_y, X.shape[1] - sizeIM[1])
    start_y = start_y[start_y <= (X.shape[1] - sizeIM[1])]
    start_y = np.unique(start_y)

    #Check for memory requirement here; break if greater than some thr.

    Xpatch = np.zeros((start_x.size * start_y.size, sizeIM[0], sizeIM[1]))
    patch_counter = 0
    for i in start_x.astype(int):
        for j in start_y.astype(int):
            Xpatch[patch_counter] = X[i: i+sizeIM[0], j: j + sizeIM[1]]
            patch_counter += 1
    return Xpatch
