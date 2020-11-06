#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:18:57 2020

@author: lordano
"""

import sys
sys.path.insert(0, '/home/lordano/SRW/env/work/srw_python')
from srwlib import *
import srwlpy as srwl 

import numpy as np
from matplotlib import pyplot as plt

import hoshi_srw as hsrw

#########################################################################
#### imputs
filename = '/media/lordano/DATA/Mestrado/siemens_article_zoom.jpg'
max_thickness = 600e-9
img_width = 200e-9

energy = 8000.0
material = 'Au'

transmission_filename = '/media/lordano/DATA/Mestrado/siemens_T{0}nm_W{1}nm_transm_E_{2}eV.txt'.format(int(max_thickness*1e9), int(img_width*1e9), int(energy))

#########################################################################


#### import image and normalize to thickness
img = hsrw.image_to_thickness(filename, max_thickness=max_thickness, rgb_channel=0, invert=True)

#### apply hard-coded correction to make it nearly binary
if(1):
    img[img < max_thickness *2/3] = 0.0
    img[img > max_thickness *2/3] = max_thickness

#### calculate complex transmission
transm = hsrw.thickness_to_transmission(img, energy, material)

#### assign properties to new variables
thickness = img
transm_real = transm[0].real
transm_imag = transm[0].imag
amplitude = transm[1].real
phase = transm[2].real
opd = transm[3].real

nx = img.shape[1]
ny = img.shape[0]

#### write transmission to file in srw array format
img_height = img_width * ny / nx
hor_grid = [-img_width/2, img_width/2, nx]
vert_grid = [-img_height/2, img_height/2, ny] 

if(1):
    hsrw.write_srw_transmission(transmission_filename, 
                                amplitude.reshape(1, ny, nx), 
                                opd.reshape(1, ny, nx), 
                                [energy, energy, 1], hor_grid, vert_grid)

if(0):
    
    transmission_array, egrid, xgrid, ygrid = hsrw.read_srw_transmission(transmission_filename)
    read_amp_array = transmission_array[0,0]
    read_opd_array = transmission_array[1,0]

    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=2, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.2)
    
    ax[0].set_title('amplitude image')
    im0 = ax[0].imshow(np.array(read_amp_array).reshape(ny,nx), cmap='gray')
    fig.colorbar(im0, ax=ax[0])
    ax[0].set_ylabel('y [um]')
    ax[0].set_xlabel('x [um]')
    
    ax[1].set_title('opd image')
    im1 = ax[1].imshow(np.array(read_opd_array).reshape(ny,nx), cmap='gray')
    fig.colorbar(im1, ax=ax[1])
    ax[1].set_xlabel('x [um]')        
    
if(1):

    ext = np.array([-img_width/2, img_width/2, -img_height/2, img_height/2])*1e6
    
    transmission_array, egrid, xgrid, ygrid = hsrw.read_srw_transmission(transmission_filename)    
    arTr = np.genfromtxt(transmission_filename)
    srw_transm = SRWLOptT(_nx=xgrid[2], _ny=ygrid[2], 
                          _rx=xgrid[1] - xgrid[0], _ry=ygrid[1] - ygrid[0], 
                          _arTr=arTr, 
                          _extTr=1, _Fx=1e+23, _Fy=1e+23, 
                          _x=0, _y=0, 
                          _ne=egrid[2], _eStart=egrid[0], _eFin=egrid[1])

    srw_amp_array = srw_transm.get_data(_typ=1, _dep=3)    
    srw_int_array = srw_transm.get_data(_typ=2, _dep=3)
    srw_opd_array = srw_transm.get_data(_typ=3, _dep=3)
    
    fig, ax = plt.subplots(figsize=(18, 4), nrows=1, ncols=3, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.2)
    
    ax[0].set_title('amplitude image')
    im0 = ax[0].imshow(np.array(srw_amp_array).reshape(ny,nx), cmap='gray', extent=ext)
    fig.colorbar(im0, ax=ax[0])
    ax[0].set_ylabel('y [um]')
    ax[0].set_xlabel('x [um]')
    
    ax[1].set_title('intensity image')
    im1 = ax[1].imshow(np.array(srw_int_array).reshape(ny,nx), cmap='gray', extent=ext)
    fig.colorbar(im1, ax=ax[1])
    ax[1].set_xlabel('x [um]')        

    ax[2].set_title('opd image')
    im2 = ax[2].imshow(np.array(srw_opd_array).reshape(ny,nx), cmap='gray', extent=ext)
    fig.colorbar(im2, ax=ax[2])
    ax[2].set_xlabel('x [um]')    

#### plot transmission
if(0):
    
    ext = np.array([-img_width/2, img_width/2, -img_height/2, img_height/2])*1e6
    
    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.2)
    
    ax[0,0].set_title('thickness image')
    im00 = ax[0,0].imshow(thickness, cmap='gray', extent=ext)
    fig.colorbar(im00, ax=ax[0,0])
    ax[0,0].set_ylabel('y [um]')
    
    ax[0,1].set_title('real part')
    im01 = ax[0,1].imshow(transm_real, cmap='gray', extent=ext)
    fig.colorbar(im01, ax=ax[0,1])
    
    
    ax[0,2].set_title('imag part')
    im02 = ax[0,2].imshow(transm_imag, cmap='gray', extent=ext)
    fig.colorbar(im02, ax=ax[0,2])
    
    
    ax[1,0].set_title('amplitude')
    im10 = ax[1,0].imshow(amplitude, cmap='gray', extent=ext)
    fig.colorbar(im10, ax=ax[1,0])
    ax[1,0].set_ylabel('y [um]')
    ax[1,0].set_xlabel('x [um]')
    
    ax[1,1].set_title('phase')
    im11 = ax[1,1].imshow(phase, cmap='gray', extent=ext)
    fig.colorbar(im11, ax=ax[1,1])
    ax[1,1].set_xlabel('x [um]')
    
    ax[1,2].set_title('OPD')
    im12 = ax[1,2].imshow(opd, cmap='gray', extent=ext)
    fig.colorbar(im12, ax=ax[1,2])
    ax[1,2].set_xlabel('x [um]')
    
    plt.savefig(transmission_filename[:-4]+'.png', dpi=600)












