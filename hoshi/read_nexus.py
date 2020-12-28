#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:33:33 2020

@author: lordano
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import hoshi.ptycho as hpty
import sys

######### INPUTS

filename = 'quarta_0007.hdf5'
output_filename = filename[:-5] + '_processed.hdf5'

detectors = ['pimega_image']
motors = ['piezo_top_x', 'piezo_top_y']

pixel_size = 55e-6
center_pixel = [617, 615]

new_shape = [192,192]
marker = -1

bad_columns = [79]

bad_lines = []

bad_pixels = [[ 99,  97],
              [102,  85],
              [101, 110],
              [101,  80],
              [100, 113],
              [121,  92],
              [111,  81],
              [ 82,  98],
              [134,  92],
              [145,  94],
              [147,  93], 
              [ 60,  92], 
              [ 27, 102],
              [ 21,  95],
              [ 14, 104],
              [  6, 100],
              [ 40,  91],
              [ 91,  63],
              [ 95,  56],
              [104,  54],
              [ 92,  53],
              [ 92,  47],
              [ 89,  46],
              [ 92,  44],
              [ 89,  46],
              [ 92,  44],
              [ 91,  43],
              [ 91,  42],
              [101,  40],
              [100,  40],
              [146,  27],
              [116,   4],
              [128,   0],
              [  2, 146],
              [ 21,  79],
              [174, 179],
              [109, 177]
              ]

######### READ DATA

detectors_list, motors_list, attributes = hpty.read_nexus_file(filename, detectors, motors)
    
x = np.array(motors_list[0]) * 1e-6
y = np.array(motors_list[1]) * 1e-6

positions = np.zeros((int(len(x)*len(y)), 2))
counter = 0
for j in range(len(y)):
    for i in range(len(x)):
        positions[counter] = np.array([x[j, i], y[j, i]])
        counter += 1

### use center value as origin

x_avg = np.average(x, axis=0)
x_center = np.mean(x_avg)
x_corrected = x_avg - x_center
positions[:,0] -= x_center

y_avg = np.average(y, axis=1)
y_center = np.mean(y_avg)
y_corrected = y_avg - y_center
positions[:,1] -= y_center

#sys.exit()

######### PROCESS AND SAVE DATA

with h5py.File(output_filename, 'w') as f:
     
    f.attrs['energy'] = attributes['energy']
    f.attrs['energy unit'] = 'eV'
    f.attrs['detector distance'] = attributes['detector_distance']
    f.attrs['distance unit'] = 'm'
    f.attrs['sample'] = 'ANT collapsed pattern'
    f.attrs['positions'] = positions
    

    diff_grp = f.create_group('diffracted intensity')
    diff_grp.attrs['xi'] = 0
    diff_grp.attrs['xf'] = new_shape[1] * pixel_size
    diff_grp.attrs['xn'] = new_shape[1]
    diff_grp.attrs['yi'] = 0
    diff_grp.attrs['yf'] = new_shape[0] * pixel_size
    diff_grp.attrs['yn'] = new_shape[0]
    diff_grp.attrs['unit'] = 'm'

    counter = 0
    for j in range(len(y)):
        for i in range(len(x)):

            # get position
            x_sample, y_sample = positions[counter]
                        
            # get diffraction pattern            
            img = detectors_list[0][i,j,:,:].astype(float)
            
            # crop diffraction pattern            
            img = hpty.crop_matrix(img, new_shape, center_pixel, 0)
            
            # mark bad pixels
            img = hpty.mark_bad_pixels(img, bad_pixels, marker)
            img = hpty.mark_bad_columns(img, bad_columns, marker)
            img = hpty.mark_bad_lines(img, bad_lines, marker)
            
            # write dataset
            diff_intensity = f['diffracted intensity'].create_dataset('intensity_{0:04d}'.format(int(counter+1)),
                                                           data=img, dtype=float, 
                                                           compression="gzip")
            
            diff_intensity.attrs['x_sample'] = x_sample
            diff_intensity.attrs['y_sample'] = y_sample    
        
            counter += 1







if(0):

    ######### CROP DIFFRACTION PATTERNS
    
    img0 = detectors_list[0][21,15,:,:].astype(float)
    img0 = hpty.crop_matrix(img0, new_shape, center_pixel, 0)
    
    ######### MARK BAD PIXELS
    
    
    img0 = hpty.mark_bad_pixels(img0, bad_pixels, marker)
    img0 = hpty.mark_bad_columns(img0, bad_columns, marker)
    img0 = hpty.mark_bad_lines(img0, bad_lines, marker)
    
    ######### SHOW DIFFRACTION PATTERNS
    
    if(0):
        vmin, vmax = img0.min(), img0.max()
        # vmin, vmax = 0, 100
        
        
        fig, ax = plt.subplots(ncols=2)
        im1 = ax[0].imshow(img0, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        im2 = ax[1].imshow(img0, origin='lower', cmap='jet', vmin=vmin+2, vmax=vmax, norm=LogNorm())
        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])





























def condition_larger_than(matrix, value=2**24):
    return matrix > value

def condition_smaller_than(matrix, value=2**24):
    return matrix < value

def get_larger_than(matrix, value):
    
    shape = matrix.shape
    matrix_flatten = matrix.reshape(int(shape[0]*shape[1]))
    
    boolean = condition_larger_than(matrix_flatten, value)
    boolean = boolean.reshape(shape)
    print(boolean)
    return np.where(boolean)

def get_smaller_than(matrix, value):
    boolean = condition_smaller_than(matrix, value)
    return np.where(boolean)[0]


