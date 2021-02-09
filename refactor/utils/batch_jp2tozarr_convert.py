#!/usr/bin/env python3

import shutil,os
import sys
import glymur #conda install -c sunpy glymur
from glob import glob
import zarr

glymur.set_option('lib.num_threads', 2)


if not os.path.exists(str(sys.argv[2])+'/csv'):
    os.makedirs(str(sys.argv[2])+'/csv')


barcode = sys.argv[1].split('-')[0]
jp2_list = glob(barcode+'-????_*/*-????.jp2')
jp2_list.sort()

for file_name in jp2_list:
    if shutil.fnmatch.fnmatch(file_name, '*-????.jp2'):
        print(file_name.split('.')[0].split('/')[-1])
        jp2 = glymur.Jp2k(file_name)
        zarr.save(str(sys.argv[2])+'/'+str(file_name.split('.')[0].split('/')[-1])+'.zarr', jp2[:])
