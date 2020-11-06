#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:49:32 2020

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
import copy

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
transmission_filename = 'siemens_600nm_transm_E_8000eV.txt'

transmission_array, egrid, xgrid, ygrid = hsrw.read_srw_transmission(transmission_filename)    
arTr = np.genfromtxt(transmission_filename)
srw_transm = SRWLOptT(_nx=xgrid[2], _ny=ygrid[2], 
                      _rx=xgrid[1] - xgrid[0], _ry=ygrid[1] - ygrid[0], 
                      _arTr=arTr, 
                      _extTr=0, _Fx=1e+23, _Fy=1e+23, 
                      _x=0, _y=0, 
                      _ne=egrid[2], _eStart=egrid[0], _eFin=egrid[1])
    
    
pp_sample = hsrw.srw_pp(0.0, 2.0, 1.0, 2.0, 1.0)

oes = []
pps = []

oes.append(srw_transm)
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

    sample_FT = np.abs(np.fft.fftshift(np.fft.fft2(beam[1:,1:])))**2
    sample_FT /= np.max(sample_FT)
    
    plt.figure()
    plt.imshow(np.log(sample_FT), origin='lower', aspect='auto', cmap='jet')
    


###### propagate through some distance


if(1):
    
    distance = 20.0e-6
    
    
    drift = SRWLOptD(distance)
    pp_drift = hsrw.srw_pp(0, 8.0, 1.0, 4.0, 1.0)
    
    optBL2 = SRWLOptC([drift], [pp_drift])
    srwl.PropagElecField(wfr, optBL2)
    
    
    beam = read_srw_wfr(wfr=wfr,
                        pol_to_extract=6,
                        int_to_extract=0)
    
    plot_beam(beam)



if(1):
    
    z = 100.0e-3
    
    
    drift = SRWLOptD(z)
    pp_drift = hsrw.srw_pp(3, 1.0, 1.0, 1.0, 1.0)
    
    optBL2 = SRWLOptC([drift], [pp_drift])
    srwl.PropagElecField(wfr, optBL2)
    
    
    beam_prop = read_srw_wfr(wfr=wfr,
                             pol_to_extract=6,
                             int_to_extract=0)
    
    beam_prop_log = copy.deepcopy(beam_prop)
    beam_prop_log[1:,1:] /= np.max(beam_prop_log[1:,1:])
    beam_prop_log[1:,1:] = np.log(beam_prop_log[1:,1:])
    
    plot_beam(beam_prop)
    plot_beam(beam_prop_log)
