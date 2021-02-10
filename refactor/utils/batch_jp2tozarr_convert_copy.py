#!/usr/bin/env python3

import shutil,os
import sys
import glymur #conda install -c sunpy glymur
glymur.set_option('lib.num_threads', 2)
import argparse
from glob import glob
import zarr

parser = argparse.ArgumentParser()
parser.add_argument("--jp2_path", type=str, help='path to example slide path, such as: /allen/programs/celltypes/production/mousegenetictools/prod34/0539049651-0003_1059789802/')
parser.add_argument("--zarr_path", type=str, help='path to where you want to store zarr files')

def main(jp2_path,zarr_path):
    if not os.path.exists(zarr_path+'/csv'):
        os.makedirs(zarr_path+'/csv')

    barcode = jp2_path.split('-')[0]
    jp2_list = glob(barcode+'-????_*/*-????.jp2')
    jp2_list.sort()

    for file_name in jp2_list:
        if shutil.fnmatch.fnmatch(file_name, '*-????.jp2'):
            print(file_name.split('.')[0].split('/')[-1])
            jp2 = glymur.Jp2k(file_name)
            zarr.save(zarr_path+'/'+str(file_name.split('.')[0].split('/')[-1])+'.zarr', jp2[:])
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))