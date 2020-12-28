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
    from pathlib import Path

    curr_path = str(Path().absolute())
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellCount/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Local/'

    #rel_im_path = 'dat/raw/Dataset_03_Images_clean/'
    #rel_lbl_path = 'dat/proc/Dataset_03_Labels_v1/'
    #rel_result_path = 'dat/results/'
    rel_im_path = '5Brain/'
    rel_lbl_path = '5Brain/'
    rel_result_path = '5Brain_proc_v2/'
    
    assert (os.path.isdir(base_path) and
           os.path.isdir(base_path + rel_im_path) and
           os.path.isdir(base_path + rel_lbl_path) and
           os.path.isdir(base_path + rel_result_path)), "Check if paths exist"

    return base_path, rel_im_path, rel_lbl_path, rel_result_path

def get_fileid():
    """Return fileids for _raw.tif images in directory.
    \n Outputs: 
    \n fid_im_lbl -- Both IM and corresponding Labels available
    \n fid_im -- IM available 
    \n fid_lbl -- Labels available"""    


    base_path, rel_im_path, rel_lbl_path = set_paths()[0:3]
    search_pattern_im = base_path + rel_im_path + '*.tif'
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