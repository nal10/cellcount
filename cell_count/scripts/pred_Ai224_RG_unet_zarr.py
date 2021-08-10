#### Prediction with trained model
# 1. Loads model, and weights
# 2. Load zarr dataset for prediction (patch generation is handled by this class)
# 3. Predict, perform checks on subtiles, and write results to .csv

from pathlib import Path

import argparse
import glob
import numpy as np
import pandas as pd
import torch
import time
import os
from cell_count.models import Ai224_RG_UNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from cell_count.utils.analysis import pred_to_xy
from cell_count.utils.data import Pred_Ai224_RG_Zarr, Pred_Sampler_Zarr
from cell_count.utils.post_processing import remove_duplicate_points

start_time = time.time()

#Torch convenience functions
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tensor = lambda x: torch.tensor(x).to(dtype=torch.float32).to(device)
tensor_ = lambda x: torch.as_tensor(x).to(dtype=torch.float32).to(device)
tonumpy = lambda x: x.cpu().detach().numpy()

parser = argparse.ArgumentParser()
parser.add_argument("--im_path",   default='/home/elyse/allen/programs/celltypes/workgroups/mct-t200/Molecular_Genetics_Daigle_Team/Elyse/Unet_WB_testing/546117/', type=str)
parser.add_argument("--csv_path",  default='/home/elyse/allen/programs/celltypes/workgroups/mct-t200/Molecular_Genetics_Daigle_Team/Elyse/Unet_WB_testing/TESTS/', type=str)


def main(im_path=None, csv_path=None):

    model = Ai224_RG_UNet()
    ckpt_file = '../../../dat/Ai224_RG_models/CE_wt244_Adam_run_v3_norm/44587_ckpt.pt'
    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']

    model.to(device)
    print(f'Validation loss was {loss:0.8f}')
    model.eval()

    patch_size = 260
    output_size = 172
    scale = 255.
    n_x_patch_per_subtile = 10
    n_y_patch_per_subtile = 10
    batch_size = n_x_patch_per_subtile*n_y_patch_per_subtile

    fname_list = glob.glob(im_path+"*.zarr")
    fname_list = sorted([Path(f).name for f in fname_list])
    for fname in fname_list:
        print(f'Segmenting {fname} ...')
        csv_fname_g = fname.replace('.zarr', '_g.csv')
        csv_fname_r = fname.replace('.zarr', '_r.csv')

        pred_dataset = Pred_Ai224_RG_Zarr(patch_size = patch_size,
                                          output_size=output_size,
                                          im_path=im_path,
                                          fname=fname,
                                          scale=scale)

        pred_sampler = Pred_Sampler_Zarr(dataset=pred_dataset,
                                         n_x_patch_per_subtile=n_x_patch_per_subtile,
                                         n_y_patch_per_subtile=n_y_patch_per_subtile)

        pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False,
                                     sampler=pred_sampler, drop_last=False, pin_memory=True)

        pred_datagen = iter(pred_dataloader)

        # empty subtile for labels:
        n_labels = 3
        x_size = pred_dataset.output_size*pred_sampler.n_x_patch_per_subtile
        y_size = pred_dataset.output_size*pred_sampler.n_y_patch_per_subtile
        output_size = pred_dataset.output_size
        offset = int((pred_dataset.patch_size - pred_dataset.output_size)/2)
        new_csv=True

        # each batch has 1 subtile.
        for _ in tqdm(range(len(pred_datagen))):
            # init empty subtile
            pred_g = np.empty(shape=[n_labels, x_size, y_size], dtype=float)
            pred_r = np.empty(shape=[n_labels, x_size, y_size], dtype=float)

            batch = next(pred_datagen)
            with torch.no_grad():
                xg, xr, _, _ = model(tensor_(batch['im']))
            xg = tonumpy(xg)
            xr = tonumpy(xr)

            subtile_ind = tonumpy(batch['idx'][0])
            for j in range(batch['idx'].shape[0]):
                ind = tonumpy(batch['idx'][j]) - subtile_ind
                pred_g[:,ind[0]:ind[0]+output_size,ind[1]:ind[1]+output_size] = xg[j]
                pred_r[:,ind[0]:ind[0]+output_size,ind[1]:ind[1]+output_size] = xr[j]

            del xg,xr,batch
            com_g,n_elem_g = pred_to_xy(fg=np.squeeze(pred_g[2,:,:]),bo=np.squeeze(pred_g[1,:,:]),pred_thr=0.5,n_elem_thr=5)
            com_r,n_elem_r = pred_to_xy(fg=np.squeeze(pred_r[2,:,:]),bo=np.squeeze(pred_r[1,:,:]),pred_thr=0.5,n_elem_thr=5)

            # convert patch co-ordinates into global co-ordinates
            global_com_g = com_g + subtile_ind.reshape(1, 2) + offset
            global_com_r = com_r + subtile_ind.reshape(1, 2) + offset

            df_g = pd.DataFrame({'x':global_com_g[:,0],'y':global_com_g[:,1],'n':n_elem_g})
            df_r = pd.DataFrame({'x':global_com_r[:,0],'y':global_com_r[:,1],'n':n_elem_r})

	        # Write to csv
            df_g = pd.DataFrame({'x':global_com_g[:,0],'y':global_com_g[:,1],'n':n_elem_g})
            df_r = pd.DataFrame({'x':global_com_r[:,0],'y':global_com_r[:,1],'n':n_elem_r})
            if new_csv:
                df_g.to_csv(csv_path+csv_fname_g, mode='w', header=True, index=False)
                df_r.to_csv(csv_path+csv_fname_r, mode='w', header=True, index=False)
                new_csv = False
            else:
                df_g.to_csv(csv_path+csv_fname_g, mode='a', header=False, index=False)
                df_r.to_csv(csv_path+csv_fname_r, mode='a', header=False, index=False)

        #write post-processed df to csv
        df_g_total = pd.read_csv(csv_path+csv_fname_g)
        df_r_total = pd.read_csv(csv_path+csv_fname_r)

        if not os.path.exists(csv_path+'processed/'):
            os.makedirs(csv_path+'processed/')
        coord_g_clean, coord_g_removed = remove_duplicate_points(df_g_total, r=10, n=50)
        df_g_clean = pd.DataFrame({'x':coord_g_clean[:,0],'y':coord_g_clean[:,1],'n':coord_g_clean[:,2]})
        df_g_removed = pd.DataFrame({'x':coord_g_removed[:,0],'y':coord_g_removed[:,1],'n':coord_g_removed[:,2]})
        df_g_clean.to_csv(csv_path+'processed/'+csv_fname_g.split('.')[0]+'_nodups.csv', header=True, index=False)
        df_g_removed.to_csv(csv_path+'processed/'+csv_fname_g.split('.')[0]+'_removed.csv', header=True, index=False)

        coord_r_clean, coord_r_removed = remove_duplicate_points(df_r_total, r=10, n=50)
        df_r_clean = pd.DataFrame({'x':coord_r_clean[:,0],'y':coord_r_clean[:,1],'n':coord_r_clean[:,2]})
        df_r_removed = pd.DataFrame({'x':coord_r_removed[:,0],'y':coord_r_removed[:,1],'n':coord_r_removed[:,2]})
        df_r_clean.to_csv(csv_path+'processed/'+csv_fname_r.split('.')[0]+'_nodups.csv', header=True, index=False)
        df_r_removed.to_csv(csv_path+'processed/'+csv_fname_r.split('.')[0]+'_removed.csv', header=True, index=False)

    print('Segmentation took {} hours'.format(round((time.time() - start_time)/3600,2)))
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
