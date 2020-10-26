#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:38:03 2020

@author: lordano
"""

import sys
sys.path.insert(0, '/home/lordano/SRW/env/work/srw_python')
from srwlib import *
import srwlpy as srwl 

import numpy as np
from matplotlib import pyplot as plt

import hoshi_srw as hsrw
from optlnls.importing import read_srw_wfr
from optlnls.plot import plot_beam


###### create plane wave

energy = 8.0e3
wavelength = 1239.84197*1e-9 / energy 

wfr = hsrw.PlaneWfr(amplitude=1e6, 
                    phase=0.5, 
                    xStart=-100e-9, 
                    xFin=100e-9, 
                    nx=400, 
                    yStart=-100e-9, 
                    yFin=100e-9, 
                    ny=200, 
                    eStart=energy, 
                    eFin=energy, 
                    ne=1)


###### create sample made of an array of slits
sample = hsrw.ApertureArraySRW(Dx=20e-9, Dy=1e-3, x0=[-50e-9, -10e-9, 70e-9])
pp_sample = hsrw.srw_pp(0.0, 2.0, 1.0, 2.0, 1.0)

oes = []
pps = []

for i in range(len(sample)):
    oes.append(sample[i])
    pps.append(pp_sample)


###### propagate wavefront to sample
optBL1 = SRWLOptC(oes, pps)
srwl.PropagElecField(wfr, optBL1)


###### extract desired intensity and plot
if(1):
    
    beam = read_srw_wfr(wfr=wfr,
                        pol_to_extract=6,
                        int_to_extract=0)
    plot_beam(beam)





###### propagate through some distance


if(0):
    
    distance = 500.0e-6
    
    
    drift = SRWLOptD(distance)
    pp_drift = hsrw.srw_pp(0, 8.0, 1.0, 4.0, 1.0)
    
    optBL2 = SRWLOptC([drift], [pp_drift])
    srwl.PropagElecField(wfr, optBL2)
    
    
    beam = read_srw_wfr(wfr=wfr,
                        pol_to_extract=6,
                        int_to_extract=0)
    
    plot_beam(beam)



if(1):
    
    z = 100.0e-6
    
    
    drift = SRWLOptD(z)
    pp_drift = hsrw.srw_pp(3, 1.0, 1.0, 1.0, 1.0)
    
    optBL2 = SRWLOptC([drift], [pp_drift])
    srwl.PropagElecField(wfr, optBL2)
    
    
    beam_prop = read_srw_wfr(wfr=wfr,
                             pol_to_extract=6,
                             int_to_extract=0)
    
    plot_beam(beam_prop)


    srw_x_coords = beam_prop[0,1:]*1e-3
    srw_x_cut = beam_prop[1:,1:][int(len(beam)/2), :]


###### get a horizontal cut and plot fourier transform


x_coords = beam[0,1:]*1e-3
x_cut = beam[1:,1:][int(len(beam)/2), :]

n = len(x_coords)
d = x_coords[1] - x_coords[0]
freq = np.fft.fftfreq(n, d)
x_cut_FT = np.abs( np.fft.fft(x_cut) )**2
x_cut_FT /= np.max(x_cut_FT)

freq *= wavelength * z 

fig, ax = plt.subplots(ncols=2)
fig.set_size_inches((12, 4))
ax[0].plot(x_coords*1e6, x_cut)
ax[1].plot(freq*1e6, x_cut_FT, '-C1')
ax[1].plot(srw_x_coords*1e6, srw_x_cut / np.max(srw_x_cut), ':C0')

plt.show() 





















