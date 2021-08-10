#### Problem description:
 - Cells labeled with red, green or both markers appear in 2-channel images
 - Degree of bleed-through across imaging channels is different for each animal
 
#### Training dataset:
 - Approximate cell centers are marked in each channel
 - Size of a single channel 2048 x 2048 image is 32.0 Mb
 - (both channels for images + labels) for a 20 images = ~1.25 Gb
 - Load it all in cpu memory for patch generation while training
 
```bash
.
├── pwgt                        # create pixel-wise ground truth from manual cell center annotations
│   ├── FM.m
│   ├── FM_script.m             # generate .mat files with output from fast marching based segmentation algorithm
│   ├── lbl_from_FM.m           # use fast marching output to create channel-wise, pixel-wise label maps
│   ├── lbl_from_FM_script.m    
│   ├── optim_centers.m         # remove duplicate annotations and optimize cell center position
│   ├── optim_centers_script.m
│   ├── rem_duplicates.m
│   ├── simple3d.m              # evaluates if pixel is simple or not (digital topology)
│   └── tifvol2mat.m
├── models 
│   └── unet.py
├── utils
│    ├── analysis.py
│    ├── data.py
│    ├── transforms.py
└── scripts 
    ├── pred_Ai224_RG_unet_zarr.py
    └── train_Ai224_RG_unet.py
```

#### Image format processing
`utils.py` contains helper functions for following format conversions:
 - `.jp2 (uint16) --> np.array (uint8)`
 - `.jp2 (uint16) --> .zarr (uint16)`

#### Notes:

 - Use `pip install -e .` to enable importing of functions as `from cell_count.utils.data import function as f`
 - trained models are here: [dropbox link](https://www.dropbox.com/sh/19qthlltaq92431/AAAlpO_fFAH5eorzfY60q3_Ja?dl=0).
 - commit `3e952`: Input is scaled, equal weights used for cross-entropy loss. 60,000 epoch experiment
 - Pixel-wise ground truth codes adapted from (topo-preserve-fastmarching repository)[https://github.com/rhngla/topo-preserve-fastmarching]

#### Prediction of cell centers on full brain images
 - Each slice takes ~`500 s` just for I/O from .zarr format. 
 - ~`150` slices per brain will require ~`1250 min` for I/O
 - Passing through GPU + morphological ops to get co-ordinates is negligible in comparison

#### References
 - [Commonly used loss functions for segmentations](https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch)


