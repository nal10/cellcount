#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:55:51 2022

@author: nicholas.lusk
"""
import os
import shutil
import requests
import warnings
import istarmap # A patch to use tqdm with multiprocessing

import allensdk.internal.core.lims_utilities as lims_utilities

from tqdm import tqdm
from itertools import repeat
from multiprocessing import get_context, freeze_support

warnings.simplefilter('always')

#===============================================================================
## Class for getting the images 
#===============================================================================

class ImageFetcher(object):
    '''Downloads and stored JPEG tissuecyte images from LIMS

    Parameters
    ----------
    top_level : str
                path to directory where images will be saved
    '''
    
    
    # location of image service on lims2. needed to download images
    IMAGE_SERVICE_STRING = 'http://lims2/cgi-bin/imageservice'

    def __init__(self, image_serial_id, downsample, quality):
        
        self.image_metadata = {}
        self.img_id = image_serial_id
        self.downsample = downsample
        self.quality = quality
        self.queries = {'img_set': 'SELECT sb.id, sb.specimen_tissue_index FROM image_series im '\
                                   'JOIN sub_images sb ON sb.image_series_id = im.id '\
                                   'WHERE im.id = \'{0}\' '\
                                   'ORDER BY specimen_tissue_index ASC'.format(image_serial_id),
                        'thickness': 'SELECT thickness FROM image_series im '\
                                     'JOIN specimens sp ON im.specimen_id = sp.id '\
                                     'JOIN specimen_blocks sb ON sp.id = sb.specimen_id '\
                                     'JOIN blocks ON sb.block_id = blocks.id '\
                                     'JOIN thicknesses th ON blocks.thickness_id = th.id '\
                                     'WHERE im.id = \'{0}\''.format(image_serial_id)}

#===============================================================================


    def fetch_image(self, index, projection=False, force_update=False):
        '''Downloads a single tissuecyte jpeg from LIMS
        
        Parameters
        ----------
        index: int
               the number within the image set that you want (same )

        '''
        
        # Gets all the image data and orders it
        sub_images = lims_utilities.query(self.queries['img_set'])[index]
        sub_id = sub_images['id']

        # Get equalization and .aff pyramid
        equalization = get_equalization(self.img_id)
        aff_path = get_aff_path(sub_id, projection)
        
        response = self.get_image_response(aff_path, equalization)
        
        return response
        
        

#==============================================================================
        
    def fetch_set(self, save_path, projection=False, force_update=False):
        ''' Download image set for flythrough video 
        
        Parameters
        ----------
        projection : bool, optional
                 whether to find a projection mask (True) or section image
                 (False, default)
        force_update : bool, optional
                       If set to True, all requested images will be downloaded.
                       If set to False (default) requests for previously
                       downloaded images will be skipped
        '''
        
        # If the path to where you are saving the files does not exist create it
        if projection:
            image_series_dir = os.path.join(save_path, str(self.img_id) + '_projection')
        else:
            image_series_dir = os.path.join(save_path, str(self.img_id))
            
        if save_path and not os.path.exists(image_series_dir):
            os.makedirs(image_series_dir)
        
        # Gets all the image data and orders it
        sub_images = lims_utilities.query(self.queries['img_set'])
        sub_images_sorted = sorted(sub_images, key = lambda x: x['specimen_tissue_index'])
        
        # Get thickness
        thickness = lims_utilities.query(self.queries['thickness'])[0]['thickness']
            
        # Loop through the images in parallel and download them 
        inputs = zip(sub_images_sorted, repeat(thickness), 
                     repeat(image_series_dir), repeat(projection))
            
        with get_context('spawn').Pool() as pool:
            for _ in tqdm(pool.istarmap(self.save_image, inputs), total = len(sub_images_sorted)):
                pass
        
        
#===============================================================================

    def save_image(self, sub_image_data, thickness, img_dir, projection = False, force_update = False):
            
            # Get metadata related to the image for file naming purposes
            sub_id, depth = sub_image_data['id'],\
                            sub_image_data['specimen_tissue_index'] * thickness
        
            image_file = 'im_{0}_depth_{1}_ds_{2}_qa_{3}'.format(self.img_id,
                                                                 depth,
                                                                 self.downsample,
                                                                 self.quality)
            # Whether the image is a projection mask
            if projection:
                image_file = image_file + '_projection.jpeg'
            else:
                image_file = image_file + '.jpeg'
                
            equalization = get_equalization(self.img_id)

            image_path = os.path.join(img_dir, image_file)
            
            # If the image file does not exist or if you want to force an overwrite
            if force_update or not os.path.exists(image_path):

                aff_path = get_aff_path(sub_id, projection)
                response = self.get_image_response(aff_path, equalization)

                with open(image_path, 'wb') as image_file:
                    shutil.copyfileobj(response.raw, image_file)
                del response

#===============================================================================

    def get_image_response(self, aff_path, equalization = None, cmap = None):
        '''Gets an image as a raw socket response

        Parameters
        ----------
        aff_path : str
                   unix (i.e. begins with '/storage/') path to .aff pyramid
        equalization : list of numeric (len 6), optional
            scale the image nicely using RL, RU, GL, GU, BL, BU

        Returns
        -------
        requests response object
            raw JPEG image
        '''

        url = r'{0}?path={1}&'\
               'downsample={2}&'\
               'quality={3}'.format(ImageFetcher.IMAGE_SERVICE_STRING,
                                    aff_path, self.downsample, self.quality)

        if equalization is not None:
            url += '&range={0},{1},{2},{3},{4},{5}'.format(*equalization)

        if cmap is not None:
            url += '&colormap={0}'.format(cmap)


        return requests.get(url, stream=True)
    
#===============================================================================


def get_aff_path(sub_image_id, projection=False):
    '''Queries for the path to a .aff pyramid

    Parameters
    ----------
    sub_image_id : int
                   the sub_image from which a pyramid is requested
    projection : bool, optional
                 whether to find a projection mask (True) or section image 
                 (False, default)

    Returns
    -------
    str
        unix path to requested aff file
    '''

    query = 'SELECT sd.storage_directory, img.zoom FROM slides sd '\
            'JOIN images img ON img.slide_id = sd.id '\
            'JOIN sub_images si ON si.image_id = img.id '\
            'WHERE si.id = \'{0}\''.format(sub_image_id)

    storage_directory = lims_utilities.query(query)[0]['storage_directory']
    aff_base = lims_utilities.query(query)[0]['zoom']

    if not projection:
        aff_file = aff_base
    else:
        aff_file = '{0}_projection.aff'.format(aff_base.split('.')[0])

    return os.path.join(storage_directory, aff_file)

#==============================================================================

def get_equalization(image_series_id):
    '''Gets the equalization values for an image series

    Returns
    -------
    list :
        red-lower, red-upper, green-lower ... values for this image series

    '''

    query = 'SELECT * from equalizations eq '\
            'JOIN image_series ims ON ims.equalization_id = eq.id '\
            'WHERE ims.id = {0}'.format(image_series_id)
            
    try:
        data = lims_utilities.query(query)[0]
        return [data['red_lower'], data['red_upper'],
                data['green_lower'], data['green_upper'],
                data['blue_lower'], data['blue_upper']]
    except:
        print('No equalization information for series: {0}'.format(image_series_id)) 
        return None
    