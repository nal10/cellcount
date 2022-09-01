#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:39:05 2022

@author: nicholas.lusk
"""

import os
import re
import argparse
import brainrender

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from brainrender.atlas import Atlas
'''

This script is used to quantify cell counts and overlap of signals from the 
Iterative_sitk_registration.py script.

'''

# personalize settings for the brainrender window
# If WHOLE_SCREEN is TRUE it tends to crash spyder
brainrender.settings.WHOLE_SCREEN = (False)

#==============================================================================

parser = argparse.ArgumentParser(description = 'Description: quantify cell count and signal overlap',
                                 epilog = 'All is well that ends.')

parser.add_argument('input_dir', type = str, help = 'root directory to where folders for csvs are')

#==============================================================================
# Various Plot Functions
#==============================================================================

def plot_bar(df, metric = 'total', channel = None, sub_region = None, iso_region = False, average = False, ratio = False):

    # fetch the file indicating the 314 non overlapping regions to avoid redundent counting
    meso_regions, isocortex = get_region_info()
    plot_df = df.drop(columns = 'Id')
    
    
    channel_dict = {'g': 'green', 'r': 'red', 'y': 'yellow'}
    
    # plots total number of cells in the color in descending order
    if isinstance(channel, str):
        
        if iso_region:
            plot_df = plot_df.loc[plot_df['Structure'].isin(isocortex['subregion'])]
            plot_df = plot_df.merge(isocortex, left_on = 'Structure', right_on = 'subregion').drop(columns = 'subregion')
        else:
            plot_df = plot_df.loc[plot_df['Structure'].isin(meso_regions)]
        
        total_counts = plot_df.groupby(['Structure', 'mouse_ID'], as_index = False)['Total'].sum()
        total_counts = total_counts.rename(columns = {'Total': 'Sum_Total'})
        
        
        channel_df = plot_df.loc[plot_df['channel'] == channel]
        channel_df = channel_df.merge(total_counts, on = ['Structure', 'mouse_ID'])
        channel_df['normalized_vol'] = channel_df['Total'] / channel_df[channel_df.filter(regex = "volume_*").columns[0]]
        channel_df['normalized_count'] = channel_df['Total'] / channel_df['Sum_Total']
        
        
        if isinstance(sub_region, type(None)) and not iso_region:
        
            palette = {'CTX': 'red', 'CNU': 'orange', 'IB': 'gold', 'MB': 'green', 
                       'HB': 'blue', 'CBX': 'purple', 'CBN': 'violet'}
            hue_metric = 'ancestor'
            
        elif isinstance(sub_region, type(None)) and iso_region:
            palette = {'Prefrontal': 'red', 'Lateral': 'orange', 'Somatomotor': 'gold', 
                       'Visual': 'green', 'Medial': 'blue', 'Auditory': 'purple'}
            hue_metric = 'parent'
        else:
            palette = None
            hue_metric = None
            channel_df = channel_df.loc[channel_df['ancestor'] == sub_region, :]
        
        raw_order = channel_df.groupby(["Structure"])["Total"].mean().sort_values(ascending = False).index
        norm_order = channel_df.groupby(["Structure"])["normalized_count"].mean().sort_values(ascending = False).index
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('Top 25 regions by expression in ' + channel_dict[channel] + ' channel')
        
        sns.barplot(x = 'Structure', y = 'Total', hue = hue_metric , data = channel_df, 
                    palette = palette, dodge = False, order = raw_order[:10], log = True, ax = ax1)
        ax1.tick_params('x', labelrotation=90)
        ax1.set_ylabel('Cell count')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        sns.barplot(x = 'Structure', y = 'normalized_count', hue = hue_metric, data = channel_df, 
                    palette = palette, dodge = False, order = norm_order[:10], log = True, ax = ax2)
        ax2.tick_params('x', labelrotation=90)    
        ax2.set_ylabel('Cell count normalized by count /n     Sst/(Sst+Slc+both)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        plt.tight_layout()
        sns.despine()
        
    if average:
        
        palette = {'g': 'green',
                   'r': 'red',
                   'y': 'gold'}  
        
        plot_df = plot_df.groupby(by = ['region', 'channel', 'mouse_ID']).sum().reset_index()
        
        fig, ax = plt.subplots()
        sns.barplot(x = 'region', y = 'Total', hue = 'channel', data = plot_df,
                         palette = palette, log = True, ci = None, ax = ax)
        ax.set_ylabel('Total cell count (log)')
        ax.set_xlabel('Brain region Types')
        
        for i in ax.containers:
            ax.bar_label(i)

        sns.despine()
        
    if ratio:
        
        gdf, ydf = plot_df.loc[plot_df['channel'] == 'g'].sort_values(by = 'Structure', ignore_index = True), \
                   plot_df.loc[plot_df['channel'] == 'y'].sort_values(by = 'Structure', ignore_index = True)
                  
        
        cols = ['Total', 'Left', 'Right']
        plot_df = (gdf[cols] + ydf[cols]) / ydf[cols]
        plot_df[['Structure', 'region']] = gdf[['Structure', 'region']]
        plot_df = plot_df.replace(np.inf, np.nan)
        
        sns.barplot(x = 'region', y = 'Total', data = plot_df)
        sns.despine()
        
    return

# have to do this as a subplot because sns.pairplot has glitches
def plot_correlations(df, pair = True, scale = True, log = False):

    plot_df = get_meso_df(df)

    if pair:
        
        paired_df = get_paired_format(plot_df, scale)
        
        if scale:
            title = "Cell totals across regions scaled to volume"
        else:
            title = "Cell totals across regions"
        
        totals = ['Total_g', 'Total_r', 'Total']
        palette = {'CTX': 'red', 'CNU': 'orange', 'IB': 'y', 'MB': 'green', 
                   'HB': 'blue', 'CBX': 'purple', 'CBN': 'violet'}
        fig, ax = plt.subplots(3, 3, sharex = True, figsize = (10, 10))
        fig.suptitle(title)
        for row_n in range(3):
            for col_n in range(3):
                if col_n == row_n:
                    sns.kdeplot(data = paired_df, x = totals[row_n], hue = 'ancestor', 
                                fill = True, alpha = 0.2, palette = palette, log_scale = log, 
                                ax = ax[row_n, col_n])
                    if row_n < 2:
                        ax[row_n, col_n].legend().remove()
                else:
                    sns.scatterplot(data = paired_df, y = totals[row_n], x = totals[col_n], 
                                    hue = 'ancestor', palette = palette, legend = False,
                                    ax = ax[row_n, col_n])
                    if log == True:
                        ax[row_n, col_n].set(xscale = "log", yscale = "log")
             
        plt.tight_layout()   
        sns.despine()

    return

# can do with df.plot() and a pivot table but this gives better flexibility
def plot_stacked_bar(df):
    
    # get the average across mice
    df = df.groupby(by = ['Structure', 'channel'], as_index = False).mean()
    
    # get total cell count for ordering 
    total_counts = df.groupby(by = 'Structure', as_index = False)['Total'].sum()
    total_counts = total_counts.rename(columns = {'Total': 'Sum_Total'})
    
    df = df.merge(total_counts, on = 'Structure')
    df_sort = df.sort_values(by = 'Sum_Total', ascending = False, ignore_index = True)
    
    palett = ['r', 'y', 'g']
    gene = ['Sst', 'Both', 'Slc']
    
    fig, ax = plt.subplots(figsize = (6, 15))
    fig.suptitle('Gene Expression by Isocortex Subregion')
    
    for c, g in zip(palett, gene):
        if c == 'r':
            plot_df = df_sort.loc[df_sort['channel'] == c, ['Structure', 'Sum_Total']]
            plot_df = plot_df.rename(columns = {'Sum_Total': 'Total'})
        elif c == 'y':
            plot_df = df_sort.loc[df_sort['channel'].isin(['y', 'g']), ['Structure', 'Total']]
            plot_df = plot_df.groupby(by = 'Structure', as_index = False, sort = False).sum()
        elif c == 'g':           
            plot_df = df_sort.loc[df_sort['channel'] == c, ['Structure', 'Total']]
            
        sns.barplot(x = 'Total', y = 'Structure', data = plot_df, 
                    color = c, label = g, ax = ax)
        
    ax.set_ylabel('Isocortex Subregion')
    ax.set_xlabel('Total Cell Count')
    ax.legend(loc = 'center right', title = 'Gene Labeled')
    sns.despine()
    

# need to figure out best way to do this
def plot_dist_from_center(df, sub_region = None):
    
    plot_df = get_meso_df(df)
    
    if not isinstance(sub_region, type(None)):
        
        plot_df = plot_df.loc[plot_df['ancestor'] == sub_region, :]
        
    
def plot_count_along_axes(df, sub_region = None):
    
    plot_df = get_meso_df(df)
    
    if not isinstance(sub_region, type(None)):
        
        plot_df = plot_df.loc[plot_df['ancestor'] == sub_region, :]
        
    # the coordinates from brainrender are x = DV, y = AP, z = ML
    # lowest x is most ventral, lowest y is most anterior  
    AP_ordered = plot_df.groupby('Structure').mean

    
    


#==============================================================================
# Preprocessing Functions
#==============================================================================

#looks at cell location for each channel to find which are the overlapping cells
def compare_location(df):
    
    green, red, yellow = df.loc[df['channel'] == 'g'], \
                         df.loc[df['channel'] == 'r'], \
                         df.loc[df['channel'] == 'y']
    
    
    # dont need, but good code for finding where overlap exists
    #df_all = yellow.merge(green.drop_duplicates(), on=['X','Y', 'Z'], 
    #                      how='left', indicator=True)
    
    return

# a common reshaping needed for plotting
def get_paired_format(df, scale):
    
    gdf, rdf, ydf = df.loc[df['channel'] == 'g'].drop(columns = ['channel']), \
                    df.loc[df['channel'] == 'r'].drop(columns = ['channel']), \
                    df.loc[df['channel'] == 'y'].drop(columns = ['channel'])
    
    gdf = gdf.iloc[:, :-2]
    rdf = rdf.iloc[:, :-2]       
    
    all_df = gdf.merge(rdf, on = ['Structure', 'region', 'mouse_ID'], suffixes = ['_g', '_r']).merge(ydf, on = ['Structure', 'region', 'mouse_ID'])
    out_df = all_df.groupby(by = ['Structure', 'ancestor'], as_index = False)['Total_g', 'Total_r', 'Total', 'volume_(um3)'].mean()
    
    if scale:
        out_df[['Total_g', 'Total_r', 'Total']] = out_df[['Total_g', 'Total_r', 'Total']].div(out_df['volume_(um3)'], axis = 0)
    
    return out_df

# scales the number of cells to cells/mm3
#structures is a dataframe with brain regions and whether they are midline or hemi regions
def get_ref_volumes(structures, ref_atlas = None,  unit = None):
    
    if isinstance(ref_atlas, type(None)):
        ref_atlas = 'allen_mouse_25um'
    
    if isinstance(unit, type(None)):
        unit = 'um'
        factor = 1
    elif unit == 'mm':
        factor = 1000**3
        
    fname = 'volume_data_' + ref_atlas + '_scale_' + unit + '.csv'
    file = os.path.join(os.getcwd(), fname)
    
    # check if df already exists
    if os.path.exists(file):
        print('Loading {0} from file...'.format(fname))
        df = pd.read_csv(file, index_col = 0)
    else:
        print('Building {0} dataframe. May take a moment...'.format(fname))
        # Iso region comes from Harris et al., 
        labels = ['Structure', 'ancestor', 'volume_(' + unit + '3)', 'AP', 'DV', 'ML']
        taxo_groups = ['CTX', 'CNU', 'IB', 'MB', 'HB', 'CBX', 'CBN']
        
        atlas = Atlas(atlas_name = ref_atlas)
        volumes = list()
    
        for index, row in structures.iterrows():
            reg, ancestors = atlas.get_region(row['Structure']), \
                             atlas.get_structure_ancestors(row['Structure'])
        
            reg_volumes = reg.mesh.volume() / factor
        
            # find the centroid of each brain region
            # for hemi regions find the deviation from midline
            if row['region'] == 'hemi':
                region_mesh = reg.mesh.points()
                # coordinates are (AP, DV, ML) and 5700 is roughly the midline
                hemi_mesh = region_mesh[region_mesh[:, 2] > 5700]
                center = list(hemi_mesh.mean(axis = 0))
            else:    
                center = list(reg.center)
    
            # get major taxogroups. if not in one put first ansector back
            try:
                g_origin = next(iter(set(ancestors) & set(taxo_groups)))
            except:
                if len(ancestors) == 0:
                    g_origin = None
                else:
                    g_origin = ancestors[-1]
                
            volumes.append([row['Structure'], g_origin, reg_volumes, float(center[0]), float(center[1]), float(center[2]) - 5700])
    
        df = pd.DataFrame(volumes, columns = labels)
        df.to_csv(file)

    return df

# get meso region data
def get_region_info(directory = '/Users/nicholas.lusk/allen/programs/celltypes/workgroups/mct-t200/ViralCore/Nick_Lusk/'):
    
    
    meso = pd.read_csv(os.path.join(directory, 'Non_overlapping_314.csv'), names = ['region'])
    isocortex = pd.read_csv(os.path.join(directory, 'isocortical-6modules.csv'))
    
    return meso['region'].tolist() , isocortex


# get a df formated for ploting
def get_meso_df(df):
    
    meso_regions, _ = get_region_info()    
    plot_df = df.drop(columns = 'Id')
    plot_df = plot_df.loc[plot_df['Structure'].isin(meso_regions)]

    volumes = get_ref_volumes(meso_regions)
    return plot_df.merge(volumes, on = 'Structure')

# get genotype information
def get_row_labels(row):
    
   if row['mouse_ID'] in ['549591', '549594']:
       gene = 'Cre'
   else:
       gene = 'Cre/Flp'

   return gene
#==============================================================================
'''
Things that I need to add
    - total difference between red and green signals
    - absolute difference between the two by region
    - the relative difference based on total signal in the region
'''

def main(input_dir, coords = False):
    
    count_df = pd.DataFrame()
    coord_df = pd.DataFrame()
    
    # turn count .csv files into dataframe
    # currently skipping the coordinate files
    for fpath in glob(os.path.join(input_dir, '*/csv/processed/registered/*.csv')):
        if os.path.exists(fpath):
            fname = fpath.split('/')[-1]
            mouse_ID = re.search('[0-9]{6}', fpath)[0]
            
            if "counts" in fname:
                region, channel = fname.split('_')[0], fname.split('_')[-1][-5]
            
                curr_df = pd.read_csv(fpath)
                curr_df.insert(loc = 2, column = 'region', value = region)
                curr_df.insert(loc = 2, column = 'mouse_ID', value = mouse_ID)
                curr_df.insert(loc = 3, column = 'channel', value = channel)
            
                count_df = count_df.append(curr_df, ignore_index=True)
                
            if "registered_coordinates" in fname and coords == True:
                channel = fname.split('_')[-1][-5]
                
                curr_df = pd.read_csv(fpath, names = ['X', 'Y', 'Z'])
                curr_df.insert(loc = 0, column = 'channel', value = channel)
                curr_df.insert(loc = 0, column = 'mouse_ID', value = mouse_ID)
                coord_df = coord_df.append(curr_df, ignore_index=True)
    
    # add additional genotype information
    count_df['Genotype'] = count_df.apply(lambda row: get_row_labels(row), axis=1)
    
    # add atlas data
    struct_df = count_df[['Structure', 'region']].groupby(by = 'Structure', as_index = False).agg(lambda x:x.value_counts().index[0])
    vol_df = get_ref_volumes(struct_df, unit = 'mm')
    count_df = count_df.merge(vol_df, on = 'Structure', how = 'left')
    
    return count_df

if __name__ == '__main__':
    #args = parser.parse_args()
    #main(**args)
    df = main('/Users/nicholas.lusk/allen/programs/celltypes/workgroups/mct-t200/ViralCore/Nick_Lusk/Unet_project/')