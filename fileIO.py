"""Contains functions to load images, labels, weights and patches."""

# Following shows requirements for the Unet
# def load_data(self):
# 		# Replace with fileIO functions
# 		im_train = []
# 		im_train_labels = []
# 		im_test = []
# 		im_test_labels = []
# 		return im_train, im_train_labels, im_test, im_test_labels

import sys
import os
import socket
import numpy as np
import skimage.io as skio
import glob
import pdb
import random

def set_paths():
    """Return the pathname of the data directory.
    \nOutputs are: base_path, rel_IM_path, rel_label_path, rel_result_path"""
    hostname = socket.gethostname()
    #print(hostname)
    if hostname == 'Fruity.local' or hostname == 'fruity.lan':
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellCount/'
        rel_IM_path = 'dat/raw/Dataset_01_Images/'
        rel_label_path = 'dat/proc/Dataset_01_Labels_v3/'
        rel_result_path = 'dat/results/'
    elif hostname == 'rohan-ai':
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellCount/'
        rel_IM_path = 'dat/raw/Dataset_01_Images/'
        rel_label_path = 'dat/proc/Dataset_01_Labels_v3/'
        rel_result_path = 'dat/results/'
    elif hostname == 'shenqin-ai':
        base_path = '/home/shenqin/Local/CellCount/'
        rel_IM_path = 'dat/raw/Dataset_01_Images/'
        rel_label_path = 'dat/proc/Dataset_01_Labels_v3/'
        rel_result_path = 'dat/results/'
    else:
        print('File paths for hostname = ' + hostname + 'not set!')

    if not(os.path.isdir(base_path) and
           os.path.isdir(base_path + rel_IM_path) and
           os.path.isdir(base_path + rel_label_path) and
           os.path.isdir(base_path + rel_result_path)):
        print('One or more of paths in set_paths() do not exist!')

    return base_path, rel_IM_path, rel_label_path, rel_result_path


def get_fileid():
    """Return fileids for _raw.tif images in directory.
    \n Outputs: 
    \n fid_IM_Labels -- Both IM and corresponding Labels available
    \n fid_IM -- IM available 
    \n fid_Labels -- Labels available"""    

    base_path, rel_IM_path, rel_label_path = set_paths()[0:3]
    search_pattern_IM = base_path + rel_IM_path + '*raw.tif'
    search_pattern_Labels = base_path + rel_label_path + '*labels.tif'

    
    full_fid_IM = glob.glob(search_pattern_IM)
    full_fid_Labels = glob.glob(search_pattern_Labels)
    #print('Searching for fileids based on pattern: ' + search_pattern_IM)
    #print("\n".join(full_fid_IM))
    #print("\n".join(full_fid_Labels))

    #List id of filenames ending with _raw.tif
    fid_IM = []
    for f in full_fid_IM:
        os.path.basename(f)
        os.path.dirname(f)
        os.path.splitext(f)
        temp = os.path.splitext(os.path.basename(f))
        temp = temp[0].split('_')
        fid_IM.append(temp[0])

    #List id of filenames ending with _labels.tif
    fid_Labels = []
    for f in full_fid_Labels:
        os.path.basename(f)
        os.path.dirname(f)
        os.path.splitext(f)
        temp = os.path.splitext(os.path.basename(f))
        temp = temp[0].split('_')
        fid_Labels.append(temp[0])

    # Get intersection of ids for which both IM and Labels exist
    fid_IM_Labels = [val for val in fid_IM if val in fid_Labels]
    return fid_IM_Labels, fid_IM, fid_Labels

def load_IM_Lbl(fileid):
    base_path, rel_IM_path, rel_label_path, _ = set_paths()
    fname_IM = base_path + rel_IM_path + fileid + '_raw.tif'
    fname_Lbl = base_path + rel_label_path + fileid + '_labels.tif'
    this_IM = skio.imread(fname_IM)
    this_IM = this_IM[:,:,0] #<---- Choosing 1st out of 3 exact copies in channels
    this_Lbl = skio.imread(fname_Lbl)
    return this_IM, this_Lbl


def getpatches_rand(im, lbl, patchsize=64, npatches=100):
    """Create patches from inputs. Patches are appended in the first dimension
    \n Assumes patchsize is even"""
    shape = np.shape(im)
    pad = round(patchsize/2)
    im_array = np.zeros([npatches,patchsize,patchsize,1])
    lbl_array = np.zeros([npatches,patchsize,patchsize,1])

    xmid = np.random.randint(pad,shape[0]-pad,size=[npatches,],dtype=int)
    ymid = np.random.randint(pad,shape[1]-pad,size=[npatches,],dtype=int)

    #Calculating indices on image and on patch for dim 0
    pxs = xmid - pad
    pxe = xmid + pad
    
    axs = np.zeros(npatches,dtype=int)
    axe = np.ones(npatches,dtype=int)*patchsize
    
    axs[pxs<0] = abs(pxs[pxs<0])
    axe[pxe>shape[0]] = patchsize - (pxe[pxe>shape[0]] - shape[0])

    pxs[pxs<0]=0
    pxe[pxe>shape[0]]=shape[0]

    #Repeating calculation for dim 1
    pys = ymid - pad
    pye = ymid + pad
    
    ays = np.zeros(npatches,dtype=int)
    aye = np.ones(npatches,dtype=int)*patchsize
    ays[pys<0] = abs(pys[pys<0])
    aye[pye>shape[1]] = patchsize - (pye[pye>shape[1]] - shape[1])
    
    pys[pys<0]=0
    pye[pye>shape[1]]=shape[1]
    for i in range(0, npatches):
        im_array[i,axs[i]:axe[i],ays[i]:aye[i],0] = im[pxs[i]:pxe[i],pys[i]:pye[i]]
        lbl_array[i,axs[i]:axe[i],ays[i]:aye[i],0] = lbl[pxs[i]:pxe[i],pys[i]:pye[i]]
    
    return im_array, lbl_array


def getpatches_randwithfg(im, lbl, patchsize=64, npatches=10, fgfrac=.5):
    """Create patches from npatches"""
    xmid_fg,xmid_fg = np.where(lbl!=0)
    ind = np.random.random_integers(0,np.size(xpos_fg),size=[npatches,1])
    for i in range(0, npatches):
        xpos[ind[i]]
        ypos[ind[i]]
    

    shape = np.shape(im)
    pad = round(patchsize/2)
    im_array = np.zeros([npatches,patchsize,patchsize,1])
    lbl_array = np.zeros([npatches,patchsize,patchsize,1])

    xmid = np.random.randint(pad,shape[0]-pad,size=[npatches,],dtype=int)
    ymid = np.random.randint(pad,shape[1]-pad,size=[npatches,],dtype=int)

    #Calculating indices on image and on patch for dim 0
    pxs = xmid - pad
    pxe = xmid + pad
    
    axs = np.zeros(npatches,dtype=int)
    axe = np.ones(npatches,dtype=int)*patchsize
    
    axs[pxs<0] = abs(pxs[pxs<0])
    axe[pxe>shape[0]] = patchsize - (pxe[pxe>shape[0]] - shape[0])

    pxs[pxs<0]=0
    pxe[pxe>shape[0]]=shape[0]

    #Repeating calculation for dim 1
    pys = ymid - pad
    pye = ymid + pad
    
    ays = np.zeros(npatches,dtype=int)
    aye = np.ones(npatches,dtype=int)*patchsize
    ays[pys<0] = abs(pys[pys<0])
    aye[pye>shape[1]] = patchsize - (pye[pye>shape[1]] - shape[1])
    
    pys[pys<0]=0
    pye[pye>shape[1]]=shape[1]
    for i in range(0, npatches):
        im_array[i,axs[i]:axe[i],ays[i]:aye[i],0] = im[pxs[i]:pxe[i],pys[i]:pye[i]]
        lbl_array[i,axs[i]:axe[i],ays[i]:aye[i],0] = lbl[pxs[i]:pxe[i],pys[i]:pye[i]]
    
    return im_array, lbl_array









# def load_IM(fileid = []):
#     """Return images in numpy array for all files listed in fileid. Assumes all files are the same """
#     base_path, rel_IM_path, rel_label_path, rel_result_path = set_paths()[0:3]
#     IM = np.array([])
        
#     # Hardcoded values below for dimensions
#     IM = np.zeros((len(fileid), 2500, 2500))
#     for idx, item in enumerate(fileid):
#         fname = base_path + rel_IM_path + item + '_raw.tif'
#         thisIM = skio.imread(fname)
#         if len(thisIM.shape) == 3:
#             print('More than 1 channel in ' + fname +': Retaining only 1st.')
#             IM[idx] = np.copy(thisIM[:, :, 0])
#         else:
#             IM[idx] = np.copy(thisIM[:, :])
#     return IM


# def load_labels(**kwargs):
#     """Return labels in numpy array for all files listed in `fileid`.
#     _label.tif files are generated with Matlab code - the original 
#     label files could not be opened with scikit-image functions

#     Keyword arguments:
#     fileid -- string containing number identifying an image file
#     """

#     base_path, _, rel_label_path = set_paths()[0:3]
#     if 'fileid' in kwargs:
#         fileid = kwargs.pop('fileid')
#     else:
#         fileid = get_fileid()

#     # Hardcoded values below for dimensions and number of channels!
#     labels = np.zeros((len(fileid), 2500, 2500))
#     for idx, item in enumerate(fileid):
#         fname = base_path + rel_label_path + item + '_labels.tif'
#         thislabel = skio.imread(fname)
#         if len(thislabel.shape) == 3:
#             labels[idx] = np.copy(thislabel[:, :, 0])
#         else:
#             labels[idx] = np.copy(thislabel[:, :])
#     return labels


# def calc_W(IM, label):
#     """Return weights in numpy array for all given Image and label.
#     IM and label should have same dimensions. Transformations to obtain
#     W are performed in 2D"""

#     W = np.zeros(IM.shape)
#     return W


# def gen_patch(X, **kwargs):
#     """Return inputs shaped to 256 x 256 as obtained by strides.
#     stride is expected as a 2 element tuple
#     """
#     if 'stride' in kwargs:
#         stride = kwargs.pop('stride')
#     else:
#         stride = (128, 128)
#     sizeIM = (256, 256)

#     maxstrides = np.zeros(2)
#     maxstrides[0] = np.ceil((X.shape[0]-sizeIM[0])/stride[0])+1
#     maxstrides[1] = np.ceil((X.shape[1]-sizeIM[1])/stride[1])+1

#     start_x = np.arange(0, maxstrides[0], 1) * stride[0]
#     start_x = np.append(start_x, X.shape[0] - sizeIM[0])
#     start_x = start_x[start_x <= (X.shape[0] - sizeIM[0])]
#     start_x = np.unique(start_x)

#     start_y = np.arange(0, maxstrides[1], 1) * stride[1]
#     start_y = np.append(start_y, X.shape[1] - sizeIM[1])
#     start_y = start_y[start_y <= (X.shape[1] - sizeIM[1])]
#     start_y = np.unique(start_y)

#     #Check for memory requirement here; break if greater than some thr.

#     Xpatch = np.zeros((start_x.size * start_y.size, sizeIM[0], sizeIM[1]))
#     patch_counter = 0
#     for i in start_x.astype(int):
#         for j in start_y.astype(int):
#             Xpatch[patch_counter] = X[i: i+sizeIM[0], j: j + sizeIM[1]]
#             patch_counter += 1
#     return Xpatch


# def stack_list(X_list):
#     '''Puts contents of X_list into a single numpy array. Arrays are stacked in along axis = 0 in output X.
#     input X_list is a 1d list. Each element is a 3d numpy array of same size along axis 1 and 2.
#     output X is a 3d numpy array.'''
    
#     n_planes = 0
#     for X in X_list:
#         n_planes += np.size(X, 0)

#     #Pre-allocate array size
#     X = np.zeros((n_planes, np.size(X_list[0], 1), np.size(X_list[0], 2)))

#     #Pop the list entries into the array
#     this_plane = int(0)

#     #Do this while X_list is not empty
#     while len(X_list) > 0:
#         curr_stacksize = np.size(X_list[0], 0)
#         X[this_plane:this_plane + curr_stacksize] = X_list.pop(0)
#         this_plane = this_plane + curr_stacksize
        
#     return X


# def gen_rand_patches(X, patchsize=(256, 256), stride=(128, 128), npatches=10, ispad=True):
#     """
#     "Creates random patches of given size from a given 2d or 3d image X.
#     \n ispad: True allows patches to be chosen from arbitrary position within image, padded with zeros.
#     \n Outputs: 
#     \n 3d array with dims (patchsize[0] x patchsize[1] x npatchs)
#     """
#     if ispad:
#         patch_mid_x = np.random.randint(0, X.shape[0], npatches)
#         patch_mid_y = np.random.randint(0, X.shape[1], npatches)
#     else:
#         xystart = np.random.randint(0, X.shape[0], 2)

#     for _ in np.arange(0,npatches)
        


#     maxstrides = np.zeros(2)
#     maxstrides[0] = np.ceil((X.shape[0]-patchsize[0])/stride[0])+1
#     maxstrides[1] = np.ceil((X.shape[1]-patchsize[1])/stride[1])+1

#     start_x = np.arange(0, maxstrides[0], 1) * stride[0]
#     start_x = np.append(start_x, X.shape[0] - patchsize[0])
#     start_x = start_x[start_x <= (X.shape[0] - patchsize[0])]
#     start_x = np.unique(start_x)

#     start_y = np.arange(0, maxstrides[1], 1) * stride[1]
#     start_y = np.append(start_y, X.shape[1] - patchsize[1])
#     start_y = start_y[start_y <= (X.shape[1] - patchsize[1])]
#     start_y = np.unique(start_y)

#     #Check for memory requirement here; break if greater than some thr.

#     Xpatch = np.zeros((start_x.size * start_y.size,
#                        patchsize[0], patchsize[1]))
#     patch_counter = 0
#     for i in start_x.astype(int):
#         for j in start_y.astype(int):
#             Xpatch[patch_counter] = X[i: i+patchsize[0], j: j + patchsize[1]]
#             patch_counter += 1

#     return Xpatch
    

