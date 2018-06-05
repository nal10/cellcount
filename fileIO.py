"""Contains functions to load images, labels, weights and patches."""

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
    \nOutputs are: base_path, rel_im_path, rel_lbl_path, rel_result_path"""
    hostname = socket.gethostname()
    #print(hostname)
    if hostname == 'Fruity.local' or hostname == 'fruity.lan':
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellCount/'
        rel_im_path = 'dat/raw/Dataset_01_Images/'
        rel_lbl_path = 'dat/proc/Dataset_01_Labels_v3/'
        rel_result_path = 'dat/results/'
    elif hostname == 'rohan-ai':
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellCount/'
        rel_im_path = 'dat/raw/Dataset_01_Images/'
        rel_lbl_path = 'dat/proc/Dataset_01_Labels_v3/'
        rel_result_path = 'dat/results/'
    elif hostname == 'shenqin-ai':
        base_path = '/home/shenqin/Local/CellCount/'
        rel_im_path = 'dat/raw/Dataset_01_Images/'
        rel_lbl_path = 'dat/proc/Dataset_01_Labels_v3/'
        rel_result_path = 'dat/results/'
    else:
        print('File paths for hostname = ' + hostname + 'not set!')

    if not(os.path.isdir(base_path) and
           os.path.isdir(base_path + rel_im_path) and
           os.path.isdir(base_path + rel_lbl_path) and
           os.path.isdir(base_path + rel_result_path)):
        print('One or more of paths in set_paths() do not exist!')

    return base_path, rel_im_path, rel_lbl_path, rel_result_path

def get_fileid():
    """Return fileids for _raw.tif images in directory.
    \n Outputs: 
    \n fid_im_lbl -- Both IM and corresponding Labels available
    \n fid_im -- IM available 
    \n fid_lbl -- Labels available"""    

    base_path, rel_im_path, rel_lbl_path = set_paths()[0:3]
    search_pattern_im = base_path + rel_im_path + '*raw.tif'
    search_pattern_lbl = base_path + rel_lbl_path + '*labels.tif'
    
    full_fid_im = glob.glob(search_pattern_im)
    full_fid_lbl = glob.glob(search_pattern_lbl)
    #print('Searching for fileids based on pattern: ' + search_pattern_IM)
    #print("\n".join(full_fid_im))
    #print("\n".join(full_fid_lbl))

    #List id of filenames ending with _raw.tif
    fid_im = []
    for f in full_fid_im:
        os.path.basename(f)
        os.path.dirname(f)
        os.path.splitext(f)
        temp = os.path.splitext(os.path.basename(f))
        temp = temp[0].split('_')
        fid_im.append(temp[0])

    #List id of filenames ending with _labels.tif
    fid_lbl = []
    for f in full_fid_lbl:
        os.path.basename(f)
        os.path.dirname(f)
        os.path.splitext(f)
        temp = os.path.splitext(os.path.basename(f))
        temp = temp[0].split('_')
        fid_lbl.append(temp[0])

    # Get intersection of ids for which both IM and Labels exist
    fid_im_lbl = [val for val in fid_im if val in fid_lbl]
    return fid_im_lbl, fid_im, fid_lbl

def load_im_lbl(fileid):
    base_path, rel_im_path, rel_lbl_path, _ = set_paths()
    fname_im = base_path + rel_im_path + fileid + '_raw.tif'
    fname_lbl = base_path + rel_lbl_path + fileid + '_labels.tif'
    im = skio.imread(fname_im)
    im = im[:,:,0] #<---- Choosing 1st out of 3 exact copies in channels
    lbl = skio.imread(fname_lbl)
    return im, lbl


def getpatches_randwithfg(im, lbl, patchsize=64, npatches=10, fgfrac=.5):
    """Create patches where at least 'fgfrac' fraction of patches have atleast one foreground pixel in them
    \n Assumes 'lbl' contains foreground = 1 and background = 0"""

    shape = np.shape(im)
    pad = int(round(patchsize/2))

    nfg = int(round(npatches*fgfrac))
    if nfg > 0:
        #Find position of all foreground pixels:
        #np.where() operation takes ~0.04 s on my cpu to processing a (2500 x 2500) array.
        #These are used to create patches where at least one pixel will be foreground
        xmid_fg, ymid_fg = np.where(lbl != 0)

        #Introduce jitter allow foreground pixel to be anywhere (not only center) within the patch
        xmid_fg = xmid_fg + np.random.randint(-(pad-1), (pad-1), np.size(xmid_fg), dtype=int)
        ymid_fg = ymid_fg + np.random.randint(-(pad-1), (pad-1), np.size(ymid_fg), dtype=int)
        ind = np.random.randint(0,np.size(xmid_fg),size=npatches,dtype=int)
        
        xmid_fg = xmid_fg[ind[0:nfg]]
        ymid_fg = ymid_fg[ind[0:nfg]]
    else:
        xmid_fg = np.array([],dtype = int)
        ymid_fg = np.array([],dtype = int)

    #Create random positions around which patches will be calculated
    xmid_rand = np.random.randint(pad,shape[0]-pad,size=[npatches-nfg,],dtype=int)
    ymid_rand = np.random.randint(pad,shape[1]-pad,size=[npatches-nfg,],dtype=int)
    
    #Concatenate patch locations
    xmid = np.concatenate((xmid_fg,xmid_rand), axis=0)
    ymid = np.concatenate((ymid_fg,ymid_rand), axis=0)
    
    #Calculating indices on image and on patch for dim 0 (x - coordinate)
    fxs = xmid - pad
    fxe = xmid + pad
    
    pxs = np.zeros(npatches,dtype=int)
    pxe = np.ones(npatches,dtype=int)*patchsize
    
    pxs[fxs<0] = abs(fxs[fxs<0])
    pxe[fxe>shape[0]] = patchsize - (fxe[fxe>shape[0]] - shape[0])

    fxs[fxs<0]=0
    fxe[fxe>shape[0]]=shape[0]

    #Repeating calculation for dim 1 (y - coordinate)
    fys = ymid - pad
    fye = ymid + pad
    
    pys = np.zeros(npatches,dtype=int)
    pye = np.ones(npatches,dtype=int)*patchsize

    pys[fys<0] = abs(fys[fys<0])
    pye[fye>shape[1]] = patchsize - (fye[fye>shape[1]] - shape[1])
    
    fys[fys<0]=0
    fye[fye>shape[1]]=shape[1]

    #Assemble patches
    im_patches = np.zeros([npatches,patchsize,patchsize,1])
    lbl_patches = np.zeros([npatches,patchsize,patchsize,1])
    for i in range(0, npatches):
        im_patches[i,pxs[i]:pxe[i],pys[i]:pye[i],0] = im[fxs[i]:fxe[i],fys[i]:fye[i]]
        lbl_patches[i,pxs[i]:pxe[i],pys[i]:pye[i],0] = lbl[fxs[i]:fxe[i],fys[i]:fye[i]]
    
    return im_patches, lbl_patches

def getpatches_strides(im, lbl, patchsize=64, stride = (32, 32),padding=True):
    """Returns im_patches and lbl_patches by striding across the image.
    \n im, lbl have the same size.
    \n stride: determines overlap between patches.
    \n padding=False will discard patches that don't fully lie within the image (assumes patchsize > stride)
    """
    shape = im.shape
    if padding:
        nstrides = np.round(np.array(shape)/np.array(stride))
    else:
        nstrides = np.round((np.array(shape)-patchsize)/np.array(stride))
    nstrides = nstrides.astype(int)
    npatches = nstrides[0]*nstrides[1]

    #Determinexy-coordinates of the patches
    fxs = np.arange(0, nstrides[0], 1) * stride[0]
    fxe = fxs + patchsize

    pxs = np.zeros(np.shape(fxs),dtype=int)
    pxe = np.ones(np.shape(fxe),dtype=int)*patchsize
    
    pxs[fxs<0] = abs(fxs[fxs<0])
    pxe[fxe>shape[0]] = patchsize - (fxe[fxe>shape[0]] - shape[0])

    fxs[fxs<0]=0
    fxe[fxe>shape[0]]=shape[0]

    #Same for y-coordinate
    fys = np.arange(0, nstrides[1], 1) * stride[1]
    fye = fys + patchsize
    
    pys = np.zeros(np.shape(fys),dtype=int)
    pye = np.ones(np.shape(fye),dtype=int)*patchsize

    pys[fys<0] = abs(fys[fys<0])
    pye[fye>shape[1]] = patchsize - (fye[fye>shape[1]] - shape[1])
    
    fys[fys<0]=0
    fye[fye>shape[1]]=shape[1]
    
    ij = 0
    im_patches = np.zeros([npatches,patchsize,patchsize,1])
    lbl_patches = np.zeros([npatches,patchsize,patchsize,1])
    for i in range(len(fxs)):
        for j in range(len(fys)):
            im_patches[ij,pxs[i]:pxe[i],pys[j]:pye[j],0] = im[fxs[i]:fxe[i],fys[j]:fye[j]]
            lbl_patches[ij,pxs[i]:pxe[i],pys[j]:pye[j],0] = lbl[fxs[i]:fxe[i],fys[j]:fye[j]]
            ij += 1
    return im_patches, lbl_patches

def rotatepatches(im, lbl, patchsize=64, stride = (32, 32),padding=True):
    """Returns im_patches and lbl_patches by striding across the image.
    \n im, lbl have the same size.
    \n stride: determines overlap between patches.
    \n padding=False will discard patches that don't fully lie within the image (assumes patchsize > stride)
    """

    nstrides = np.zeros(2)
    if padding:
        nstrides = np.round(np.array(im.shape)/np.array(stride))
    else:
        nstrides = np.round((np.array(im.shape)-patchsize)/np.array(stride))

    pdb.set_trace()
    fxs = np.arange(0, nstrides[0], 1) * stride[0]
    fxs = np.append(fxs, im.shape[0] - patchsize)
    fxs = fxs[fxs <= (im.shape[0] - patchsize)]
    fxs = np.unique(fxs)

    fys = np.arange(0, nstrides[1], 1) * stride[1]
    fys = np.append(fys, im.shape[1] - patchsize)
    fys = fys[fys <= (im.shape[1] - patchsize)]
    fys = np.unique(fys)
    
    ij = 0
    im_patches  = np.zeros((fxs.size * fys.size, patchsize, patchsize))
    lbl_patches = np.zeros((fxs.size * fys.size, patchsize, patchsize))
    for i in fxs.astype(int):
        for j in fys.astype(int):
            im_patches[ij]  =  im[i: i + patchsize, j: j + patchsize]
            lbl_patches[ij] = lbl[i: i + patchsize, j: j + patchsize]
            ij += 1
    return im_patches, lbl_patches