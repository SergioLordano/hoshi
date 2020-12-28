# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:47:02 2018

@author: Sergio
"""

import numpy as np
from matplotlib import pyplot as plt
import time
import h5py
import copy

import hoshi.ptycho as hpty
    

startTime = time.time()
###############################################################################
########    USER CONTROL    ###################################################
###############################################################################

ptycho_data_filename = '/home/lordano/ptycho_measured/quarta_0007_processed.hdf5'

output_filename = ptycho_data_filename[:-5] + '_reconstruction2.h5'

d = hpty.loadDiffPatterns(ptycho_data_filename)

hpty.plot_diffraction_pattern(d, index=1)

if(1):
    
    x_obj = d['x_obj']
    y_obj = d['y_obj']
    x_ill = d['x_ill']
    y_ill = d['y_ill']
    
    obj_shape = (len(y_obj), len(x_obj)) # (ny, nx)
    illum_shape = (len(y_ill), len(x_ill)) # (ny, nx)
    
    diffPatterns = d['diff int']
    obj_positions = d['obj_positions']
    
    n_patterns = len(diffPatterns)
    
    obj_recons = hpty.uniformField(obj_shape) # Create initial guess for object - uniform
    illum_recons = hpty.makeCircAperture(15,illum_shape) # Create initial guess for illumination - uniform
    
    obj_grid = [x_obj.min(), x_obj.max(), y_obj.min(), y_obj.max()]
    illum_grid = [x_ill.min(), x_ill.max(), y_ill.min(), y_ill.max()]
    
    hpty.show_field_image(obj_recons, ext=obj_grid)
    hpty.show_field_image(illum_recons, ext=illum_grid)
    
    if(1):
        n_iterations = 200
        print('Starting reconstruction algorithm with {0:d} iterations'.format(n_iterations))
        progress = np.linspace(0, n_iterations, 21)
        count=1
        for i in range(n_iterations):
            updateProbe = 0 if i < n_iterations/5 else 1
            obj_recons, illum_recons = hpty.run_ePIE_interation(diffPatterns, obj_recons, illum_recons, 
                                                                obj_positions, updateProbe, marker=-1)
            if(i >= progress[count]):
                print("finished iteration {0} out of {1}".format(i, n_iterations))
                count += 1 
                hpty.show_field_image(obj_recons, ext=[x_obj.min(), x_obj.max(), y_obj.min(), y_obj.max()])
                hpty.show_field_image(illum_recons, ext=[x_ill.min(), x_ill.max(), y_ill.min(), y_ill.max()])
    
        print("N POINTS = ", n_patterns)
        # print("OVERLAPPING ", overlapping)
        print("N ITERATIONS = {0}".format(n_iterations))
        print("ELAPSED TIME = {0:.1f} minutes".format((time.time() - startTime)/60.0))
    
    hpty.show_field_image(obj_recons, ext=[x_obj.min(), x_obj.max(), y_obj.min(), y_obj.max()])
    hpty.show_field_image(illum_recons, ext=[x_ill.min(), x_ill.max(), y_ill.min(), y_ill.max()])
    
    hpty.save_output(output_filename, obj_recons, illum_recons, obj_grid, illum_grid)






