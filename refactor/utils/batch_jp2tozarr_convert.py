#!/usr/bin/env python3

import os
import sys
import glymur #conda install -c sunpy glymur
import zarr
glymur.set_option('lib.num_threads', 2)

for file in os.listdir(str(sys.argv[1])):
    if file.endswith(".jp2"):
        print(file.split('.')[0])
        jp2 = glymur.Jp2k(file)
        zarr.save(str(sys.argv[1])+str(file.split('.')[0])+'.zarr', jp2[:])
