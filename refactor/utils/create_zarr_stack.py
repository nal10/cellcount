import napari
import numpy
import dask.array as da
from dask import compute, delayed
import zarr
import skimage.io
from glob import glob
import pandas as pd
##testing

def create_downsampled_stack(img_id,img_subdir,save_dir,channel,slide_step=1,xy_subsample=2, x_pad=5000, y_pad=5000):
    '''A function to downsample image slides and segmented masks (this helps with visualization and QC). The stacked downsampled image and segmented masks are stored on disk. 
    
    Attributes
    ----------
    img_id : int
        image_series_id of the experiment
        
    img_subdir : str
        subdir where all the image slides are stored
        
    save_dir: dict
        directory to save stacked downsampled image and segmented masks
        
    channel: int
        channel index for image slides. Use -1 for segmented masks
        
    slide_step: int
        step size for sampling slides
    
    xy_subsample: int
        inplane subsampling factor 
        
    x_pad: int
        padding to be cropped out along x-axis (to further reduce the image size)
        
    y_pad: int
        padding to be cropped out along y-axis (to further reduce the image size)
        
    
    '''
    
    # read DN predictions
    filenames = sorted(glob("{}/*.zarr".format(img_subdir)))
    slide_ids = [f.rsplit('/',1)[1] for f in filenames ]
    
    sample = da.from_zarr(filenames[0])

    lazy_imread = delayed(da.from_zarr)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype) 
        for delayed_reader in lazy_arrays
    ]
    
    # Stack into one large dask.array

    image_stack = da.stack(dask_arrays, axis=0)[::slide_step,
                                                x_pad:-x_pad:xy_subsample,
                                                y_pad:-y_pad:xy_subsample,channel]
    zarr_save_path = '{}{}_downsampled.zarr'.format(save_dir,img_id)

    
    da.to_zarr(image_stack,zarr_save_path)
    
    print('\nDownsampled zarr stack shape: {}'.format(image_stack.shape))
    print('Saving downsampled zarr stack here: {}'.format(zarr_save_path)) 
    
    csv_save_path = '{}{}_slide_id_order.csv'.format(save_dir,img_id)
    df = pd.DataFrame()
    df['slide_ids'] = slide_ids
    df.to_csv(csv_save_path)
    print('\nSaving slide id list here: {}'.format(csv_save_path)) 

zarr_path ='/home/elyse/allen/programs/celltypes/workgroups/mct-t200/Molecular_Genetics_Daigle_Team/Elyse/Unet_WB_testing/547656/'
save_path ='/home/elyse/allen/programs/celltypes/workgroups/mct-t200/Molecular_Genetics_Daigle_Team/Elyse/Unet_WB_testing/547656/stack/'


create_downsampled_stack(547656, zarr_path, save_path, 1)
