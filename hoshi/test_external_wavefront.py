#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:50:06 2021

@author: lordano
"""
#%% import stuff
import sys
sys.path.insert(0, '/home/lordano/Software/SRW_py38/env/work/srw_python/')

import srwlpy as srwl 
from array import array

import numpy as np
from matplotlib import pyplot as plt

import h5py
from hoshi_srw import ExternalWfr
from srw_beam_caustic import SRW_beam_caustic
from srw_uti_caustic_plot import plot_caustic2D_cut, plot_caustic_slice
from optlnls.plot import plot_beam
from optlnls.importing import read_srw_wfr


#%% read amplitude and phase from reconstrucion

filename  = '/media/lordano/DATA/Mestrado/SRW/thesis_reconstructions/binning_comparison/'
filename += 'TAR_8keV_siemens_ptycho_data7x7_SS20um_pinhole_250nm_recon_D.h5'

with h5py.File(filename, 'r') as f:
    
    beam = f['illumination']
    amp = beam['amplitude'][()]
    phase = beam['phase'][()]
    mesh = beam.attrs['mesh']

#%% create srw wavefront

xStart = mesh[0]
xFin = mesh[1]
yStart = mesh[2]
yFin = mesh[3]
ny, nx = amp.shape
energy = 8000

wfr = ExternalWfr(amplitude=amp, 
                  phase=phase, 
                  xStart=xStart, 
                  xFin=xFin, 
                  nx=nx, 
                  yStart=yStart, 
                  yFin=yFin, 
                  ny=ny, 
                  eStart=energy, 
                  eFin=energy, 
                  ne=1)

#%% plot wavefront

intensity = read_srw_wfr(wfr=wfr, pol_to_extract=0, int_to_extract=0)
phase = read_srw_wfr(wfr=wfr, pol_to_extract=0, int_to_extract=4)

plot_beam(intensity)
plot_beam(phase)

#%% do beam caustic

output = 'srw_caustic.h5'

if(0):

    SRW_beam_caustic(wfr=wfr, 
                     zStart=-500e-6, 
                     zFin=+500e-6, 
                     zStep=20e-6, 
                     zOffset=0.0, 
                     extract_parameter=1, 
                     useMPI=False, 
                     save_hdf5=True, 
                     h5_filename=output, 
                     buffer=False, 
                     matrix=True, 
                     ppFin=None)

#%% plot caustic

if(1):
    
    plot_caustic2D_cut(filename_h5=output, 
                       show_axis='x', 
                       fixed_position=0.0, 
                       unitFactor=[1.0e3, 1e6], 
                       unitLabel=['Z [mm]', 'X [$\mu m$]'], 
                       aspect='auto', 
                       xlim=None, 
                       ylim=None, 
                       zlim=None, 
                       scale='linear', 
                       figsize=(8,6), 
                       fontsize=12)


    plot_caustic2D_cut(filename_h5=output, 
                       show_axis='x', 
                       fixed_position=0.0, 
                       unitFactor=[1.0e3, 1e6], 
                       unitLabel=['Z [mm]', 'X [$\mu m$]'], 
                       aspect='auto', 
                       xlim=None, 
                       ylim=None, 
                       zlim=None, 
                       scale='log', 
                       figsize=(8,6), 
                       fontsize=12)

#%% plot slice

    plot_caustic_slice(filename=output, 
                       z=0.3e-3, 
                       dataset_prefix='step_', 
                       plot='cut',
                       unitFactor=1e6, 
                       xlabel='X', 
                       ylabel='Y', 
                       units='[$\mu m$]', 
                       xlim=None, 
                       ylim=None, 
                       scale='linear', 
                       showPlot=True, 
                       showFWHM=True)






