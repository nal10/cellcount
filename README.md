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
├── pwgt                          # create pixel-wise ground truth from manual cell center annotations
│   ├── script_00_optim_centers.m # remove duplicate annotations and optimize cell center position
│   ├── script_01_FM.m            # generate .mat files with output from fast marching based segmentation algorithm
│   ├── script_02_lbl_from_FM.m   # use fast marching output to create channel-wise, pixel-wise label maps  
│   ├── FM.m                      # fast marching core - output is a .mat file per image tile
│   ├── lbl_from_FM.m             # create pixelwise labels from S, T, D maps generated by fast marching
│   ├── optim_centers.m           # optimization routine to re-position manual annotations
│   ├── rem_duplicates.m          # remove redundant annotations for the same cell
│   ├── simple3d.m                # evaluates if pixel is simple or not (digital topology)
│   └── tifvol2mat.m
├── models.py
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
 - commit `0172c`: model was trained on `control_retraining` dataset (in addition to the initial dataset)
 - trained models for this are in: [/control_retraining_combined](https://www.dropbox.com/sh/19qthlltaq92431/AAAlpO_fFAH5eorzfY60q3_Ja?dl=0). evaluate `28960_ckpt.pt` and `45000_ckpt.pt`

#### Prediction of cell centers on full brain images
 - Each slice takes ~`500 s` just for I/O from .zarr format. 
 - ~`150` slices per brain will require ~`1250 min` for I/O
 - Passing through GPU + morphological ops to get co-ordinates is negligible in comparison

#### References
 - [Commonly used loss functions for segmentations](https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch)


