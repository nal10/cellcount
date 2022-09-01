#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:08:51 2022

@author: nicholas.lusk
"""

import os
import re
import cv2
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from PIL import Image
from LIMS_jpeg import ImageFetcher
from sklearn.neighbors import KDTree

# need to update to deal with large image size
Image.MAX_IMAGE_PIXELS = 1200000000

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)


#==============================================================================
def identify_yellow_points(coord_r, coord_g, dist=8):
    red_zeroes = np.zeros((coord_r.shape[0],1))
    green_ones = np.ones((coord_g.shape[0],1))

    #0 for red, 1 for green
    coord_r = np.append(coord_r, red_zeroes, axis=1)
    coord_g = np.append(coord_g, green_ones, axis=1)
    coord_r = np.delete(coord_r, (0), axis=0)
    coord_g = np.delete(coord_g, (0), axis=0)
    coord_full = np.append(coord_r, coord_g, axis=0)

    kdt = KDTree(coord_full[:,:2], leaf_size=20, metric='euclidean')
    close_red_dist, close_red_ind = kdt.query(coord_full[:len(coord_r),:2], k=2)
    close_red_ind = np.sort(close_red_ind)

    close_indices = []
    for i, j in zip(close_red_ind, close_red_dist):
        if j[1]<=dist:
            if not list(i) in close_indices:
                coord_full[i[0]][3] = 2
                coord_full[i[1]][3] = 2

    red_cells = pd.DataFrame(coord_full[coord_full[:, 3] == 0][:,:3], columns = ['x', 'y', 'n'])
    green_cells = pd.DataFrame(coord_full[coord_full[:, 3] == 1][:,:3], columns = ['x', 'y', 'n'])
    yellow_cells = pd.DataFrame(coord_full[coord_full[:, 3] == 2][:,:3], columns = ['x', 'y', 'n'])
    
    red_cells['c'] = 'r'
    green_cells['c'] = 'g'
    yellow_cells['c'] = 'y'
    

    return pd.concat((red_cells, green_cells, yellow_cells), ignore_index = True)

#==============================================================================

def annotate_jpeg(img_id, jpeg_list, df_g, df_r, out_dir = None):
    
    if isinstance(out_dir, type(None)):
        out_dir = os.path.join(os.getcwd(), str(img_id), 'annotated')
        
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    for c, jpeg in tqdm(enumerate(jpeg_list), total = len(jpeg_list)):
        
        coord_g = df_g.loc[df_g['slide'] == c + 1, ['x', 'y', 'n']]
        coord_r = df_r.loc[df_r['slide'] == c + 1, ['x', 'y', 'n']]
        
        coord_df = identify_yellow_points(coord_r, coord_g)
        
        # work around because CV2 pixel limit is hard to modify
        image = Image.open(jpeg)
        image = np.array(image)
        
        # cv2 used BGR so need to rotate axis
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for idx in coord_df.index:
            x, y, c = coord_df.loc[idx, ['y', 'x', 'c']] # x# x and y are flipped because PIL goes by (x,y) and numpy by (row, col)
            
            if c == 'r':
                color = (0, 0, 255)
            elif c == 'g':
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            
            image = cv2.circle(image,(int(x),int(y)),25, color, 3)
            
        fname = str(img_id) + re.search('_depth_[0-9]+', jpeg).group() + '_annotated.jpeg'
        save_path = out_dir + '/' + fname
            
        cv2.imwrite(save_path, image)
        
#==============================================================================

def get_intensities(jpeg_list, df_g, df_r):
    
    save_root = os.path.split(jpeg_list[0])[0]
    save_path = os.path.join(save_root, 'intensities/')
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
            
    for c, jpeg in tqdm(enumerate(jpeg_list), total = len(jpeg_list)):
            
        coord_g = df_g.loc[df_g['slide'] == c + 1, ['x', 'y', 'n']]
        coord_r = df_r.loc[df_r['slide'] == c + 1, ['x', 'y', 'n']]
        
        coord_df = identify_yellow_points(coord_r, coord_g)
        coord_df = coord_df.loc[coord_df['c'] == 'y', ['x', 'y', 'n']]
        
        # work around because CV2 pixel limit is hard to modify
        image = Image.open(jpeg)
        
        # create list for storing data
        intensity_data = []
        pad = 100
        
        # create a different mask for each color per image
        for color in ['r', 'g']:
            
            if color == 'r':
                channel_df = pd.merge(coord_r, coord_df, on = ['x', 'y', 'n'], how='left', indicator='merge')
                channel_df['merge'] = np.where(channel_df['merge'] == 'both', True, False)
                img_channel = 2
            else:
                channel_df = pd.merge(coord_g, coord_df, on = ['x', 'y', 'n'], how='left', indicator='merge')
                channel_df['merge'] = np.where(channel_df['merge'] == 'both', True, False)
                img_channel = 1
            
            
            for idx in channel_df.index:
                
                # get cell location
                x, y, n, merge = channel_df.loc[idx, ['y', 'x', 'n', 'merge']] # x and y are flipped because PIL goes by (x,y) and numpy by (row, col)
                
                # get radius of cell for intensity measurements
                # most cells aren't perfect circles but this works
                radius = int((n / np.pi)**0.5)
                
                # only load subsection of image around cell
                left, right, top, bottom = max(0, x - pad), min(image.size[0], x + pad), \
                                           max(0, y - pad), min(image.size[1], y + pad)
                
                
                im_crop = image.crop((int(left), int(top), int(right), int(bottom)))
                image_crop = np.array(im_crop)
            
                # cv2 used BGR so need to rotate axis
                image_BGR = cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR)
                
                # crate mask of non-identified space
                mask = np.zeros(image_BGR.shape[:2], dtype="uint8")
                              
                mask = cv2.circle(mask,(int(pad),int(pad)), radius, 255, -1)
                masked = cv2.bitwise_and(image_BGR, image_BGR, mask = mask)
                
                total, avg = np.sum(masked[:, :,img_channel]), \
                             np.sum(masked[:, :, img_channel]) / n
                
                intensity_data.append((y, x, n, total, avg, color, merge))
        
        fname = re.search('depth_[0-9]+', jpeg).group() + '.csv'   
        
        out_df = pd.DataFrame(intensity_data, columns = ['x', 'y', 'n', 'total', 'average', 'channel', 'merge']) # labels are flipped back to their original from importing
        out_df.to_csv(os.path.join(save_path, fname))

#==============================================================================

def load_csv_intensities(path):
    
    csv_files = glob(os.path.join(path, '*.csv'))    
    csv_files = sorted(csv_files, key=lambda x: int(re.search('(?<=depth_)[0-9]+', x).group()))
    
    df = pd.concat([pd.read_csv(file) for file in csv_files]).reset_index()
    
    return df

#==============================================================================

def plot_intensities(df):
    
    sns.set_palette('muted')
    
    fig, axs = plt.subplots(3, 4, figsize = (12, 12), 
                            gridspec_kw = {'width_ratios':[5,1,5,1]})
    
    # top 4 plots are all red and all green cells by average intensity and cell size
    sns.kdeplot(x = 'average', data = df, hue = 'channel', shade = True, alpha = 0.3, 
                palette = {'r': 'red', 'g': 'green'}, common_norm = False, ax = axs[0, 0])
    axs[0, 0].legend(loc = 'upper center', labels = ['Sst', 'Slc'], title = 'Gene Expression',
                     fontsize = 10)
    
    sns.boxplot(x = 'channel', y = 'average', data = df, palette = {'r': 'red', 'g': 'green'},
                showfliers = False, ax = axs[0, 1])
    axs[0, 1].set_xticklabels(rotation = 30, labels = ['Sst', 'Slc'])
    
    sns.kdeplot(x = 'n', data = df, hue = 'channel', shade = True, alpha = 0.3, 
                palette = {'r': 'red', 'g': 'green'}, common_norm = False, ax = axs[0, 2])
    axs[0, 2].legend(loc = 'upper left', labels = ['Sst', 'Slc'], title = 'Gene Expression',
                     fontsize = 10)
    axs[0, 2].set_xlabel('Size of cell (px)')
    axs[0, 2].set_xlim([0, 300])
    
    sns.boxplot(x = 'channel', y = 'n', data =df, palette = {'r': 'red', 'g': 'green'}, 
                showfliers = False, ax = axs[0, 3])
    axs[0, 3].set_xticklabels(rotation = 30, labels = ['Sst', 'Slc'])
    axs[0, 3].set_ylabel('Size of cell (px)')
    
    # middle 4 plots are comparing the expression and size of red and merged cells
    sns.kdeplot(x = 'average', data = df.loc[df['channel'] == 'r', :], hue = 'merge', shade = True, alpha = 0.3, 
                palette = {True: 'hotpink', False: 'red'}, common_norm = False, ax = axs[1, 0])
    axs[1, 0].legend(loc = 'upper left', labels = ['Sst/Slc', 'Sst'], title = 'Gene Expression',
                     fontsize = 10)
    
    sns.boxplot(x = 'merge', y = 'average', data = df.loc[df['channel'] == 'r', :], 
                palette = {True: 'hotpink', False: 'red'}, showfliers = False, ax = axs[1, 1])
    axs[1, 1].set_xticklabels(rotation = 30, labels = ['Sst', 'Sst/Slc'])
    
    sns.kdeplot(x = 'n', data = df.loc[df['channel'] == 'r', :], hue = 'merge', shade = True, alpha = 0.3, 
                palette = {True: 'hotpink', False: 'red'}, common_norm = False, ax = axs[1, 2])
    axs[1, 2].legend(loc = 'upper left', labels = ['Sst/Slc', 'Sst'], title = 'Gene Expression',
                     fontsize = 10)
    axs[1, 2].set_xlabel('Size of cell (px)')
    axs[1, 2].set_xlim([0, 300])
    
    sns.boxplot(x = 'merge', y = 'n', data = df.loc[df['channel'] == 'r', :], 
                palette = {True: 'hotpink', False: 'red'}, showfliers = False, ax = axs[1, 3])
    axs[1, 3].set_xticklabels(rotation = 30, labels = ['Sst', 'Sst/Slc'])
    axs[1, 3].set_ylabel('Size of cell (px)')
    
    # middle 4 plots are comparing the expression and size of green and merged cells
    sns.kdeplot(x = 'average', data = df.loc[df['channel'] == 'g', :], hue = 'merge', shade = True, alpha = 0.3, 
                palette = {True: 'springgreen', False: 'green'}, common_norm = False, ax = axs[2, 0])
    axs[2, 0].legend(loc = 'upper right', labels = ['Sst/Slc', 'Slc'], title = 'Gene Expression',
                     fontsize = 10)
    
    sns.boxplot(x = 'merge', y = 'average', data = df.loc[df['channel'] == 'g', :], 
                palette = {True: 'springgreen', False: 'green'}, showfliers = False, ax = axs[2, 1])
    axs[2, 1].set_xticklabels(rotation = 30, labels = ['Slc', 'Sst/Slc'])
    
    sns.kdeplot(x = 'n', data = df.loc[df['channel'] == 'g', :], hue = 'merge', shade = True, alpha = 0.3, 
                palette = {True: 'springgreen', False: 'green'}, common_norm = False, ax = axs[2, 2])
    axs[2, 2].legend(loc = 'upper left', labels = ['Sst/Slc', 'Slc'], title = 'Gene Expression',
                     fontsize = 10)
    axs[2, 2].set_xlabel('Size of cell (px)')
    axs[2, 2].set_xlim([0, 300])
    
    sns.boxplot(x = 'merge', y = 'n', data = df.loc[df['channel'] == 'g', :], 
                palette = {True: 'springgreen', False: 'green'}, showfliers = False, ax = axs[2, 3])
    axs[2, 3].set_xticklabels(rotation = 30, labels = ['Slc', 'Sst/Slc'])
    axs[2, 3].set_ylabel('Size of cell (px)')
    
    sns.despine() 
    plt.tight_layout()
    

#==============================================================================

def plot_size_correlations(df):
    
    fig, ax = plt.subplots()
    
    sns.scatterplot(x = 'n', y = 'average', data = df, hue = 'channel', 
                    alpha = 0.3, ax = ax)
    ax.set_xlabel('Cell size (px)')
    ax.set_ylabel('Average pixel intensity')
    
    sns.despine()
    plt.tight_layout()

#==============================================================================
# Main Script
#==============================================================================

def main():
    
    img_id = 1068618257
    root = '/Users/nicholas.lusk'

    coord_path = root + '/allen/programs/celltypes/workgroups/mct-t200/ViralCore/Nick_Lusk/Unet_project/547656/csv/processed/'
    jpeg_path = root + '/Documents/Scripts/Nick_Scripts/Unet/Validate/' + str(img_id)
    coord_files = sorted(glob(coord_path + '*nodups.csv'))
    
    df_g, df_r = pd.DataFrame(), pd.DataFrame()
    
    # loop through the raw data and concatinate based on slide
    for file in coord_files:
        if os.path.exists(file):
            fname = file.split('/')[-1]
            slide, channel = fname.split('_')[0].split('-')[1], fname.split('_')[1]
            
            if channel == 'g':
                curr_g = pd.read_csv(file, skiprows=1)
                curr_g['slide'] = int(slide)
                df_g = df_g.append(curr_g, ignore_index = True)
                
            if channel == 'r':
                curr_r = pd.read_csv(file, skiprows=1)
                curr_r['slide'] = int(slide)
                df_r = df_r.append(curr_r, ignore_index = True)
     
    # get corresponding jpeg files
    jpeg_files = glob(jpeg_path + '/*.jpeg')
    
    # check that jpegs exists
    if len(jpeg_files) == 0:
        print('No .jpeg files meed to fetch...')
        fetch = ImageFetcher(img_id, 0, 100)
        fetch.fetch_set(os.path.split(jpeg_path)[0])
        jpeg_files = glob(jpeg_path + '/*.jpeg')
    
    jpeg_files = sorted(jpeg_files, key=lambda x: int(re.search('(?<=depth_)[0-9]+', x).group()))
    
    get_intensities(jpeg_files, df_g, df_r)
    
    
    
if __name__ == "__main__":
    main()





    

   
    
    
    

    
    


















