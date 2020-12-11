#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:31:02 2020

@author: lordano
"""
import numpy as np
from matplotlib import pyplot as plt
import time
import h5py


def randomField(shape): # For initial guesses
    amp = np.random.random(shape)
    ph = np.random.random(shape)
    ph = ph - (np.max(ph) - np.min(ph))/2
    return np.array(amp * np.exp(1j * ph))

def uniformField(shape): # For initial guesses
    amp = np.ones(shape, dtype=float)
    ph = np.ones(shape, dtype=float)
    #ph = ph - (np.max(ph) - np.min(ph))/2
    return amp * np.exp(1j * ph)

def makeCircAperture(Diameter,imagSize, amp=1.0, plot=False):
    """
    Creates a flat circular aperture
    """    
    Raio = Diameter/2.0
    x = np.arange(-(imagSize[1]-1.0)/2 , (imagSize[1]+1.0)/2, 1.0)
    y = np.arange(-(imagSize[0]-1.0)/2 , (imagSize[0]+1.0)/2, 1.0)
    xx, yy = np.meshgrid(x, y, sparse=True)
    aperture = np.zeros(imagSize,dtype=np.complex64)
    aperture[xx**2 + yy**2 < Raio**2] = amp #Circle equation
    
    if plot:
#        from matplotlib.pyplot import imshow
        plt.imshow(aperture.real)
        
    return aperture

def updateIllum(obj, illum, exitWave, exitWave_forcedAmp):
    beta = 0.5
    illum_updated = illum + beta * ( np.conjugate(obj) / (np.max(np.abs(obj))**2 + 1e-5 ) ) * (exitWave_forcedAmp - exitWave) 
    return illum_updated
    
def updateObj(obj, illum, exitWave, exitWave_forcedAmp):
    alpha = 0.9
    obj_updated = obj + alpha * ( np.conjugate(illum) / (np.max(np.abs(illum))**2 + 1e-5) ) * (exitWave_forcedAmp - exitWave) 
    return obj_updated

def run_local_iteration(obj, illum, diffPattern, updateProbe=1):
    
    exitWave = obj * illum # create exit wave
    F_exitWave = np.fft.fft2(exitWave) # fourier transform
    
    # replace modulus
    F_exitWave_abs = np.abs(F_exitWave) # get exit waves amplitudes
    F_exitWave_abs[F_exitWave_abs==0] = 1.0 # avoid division by zero
    F_exitWave = F_exitWave / F_exitWave_abs # normalize the amplitudes
    # IS IT NEEDED?? (line below)
    # F_exitWave[np.abs(F_exitWave)==0] = 1.0 # avoid zeros in the amplitudes 
    F_exitWave = F_exitWave * np.sqrt(diffPattern) # replace modulus by diffraction intensity
    
    exitWave_forcedAmp = np.fft.ifft2(F_exitWave) # inverse transform
 
    obj = updateObj(obj, illum, exitWave, exitWave_forcedAmp) # update object
    
    if(updateProbe):
        illum = updateIllum(obj, illum, exitWave, exitWave_forcedAmp) # update probe 
    
    return obj, illum

def run_ePIE_interation(diffPatterns, obj_recons, illum_recons, obj_positions, updateProbe=1):
    
    positionsIndex = np.random.permutation(len(diffPatterns)) # select positions randomly
    # print(positionsIndex)
    
    for i in positionsIndex: # random order
                
        obj_pos = obj_positions[i] # object positions for this step
        
        local_obj = obj_recons[obj_pos[0]:obj_pos[1], obj_pos[2]:obj_pos[3]] # select object ROI
        local_illum = illum_recons # select illumination ROI
        
        updated_local_obj, updated_local_illum = run_local_iteration(local_obj, 
                                                                     local_illum, 
                                                                     diffPatterns[i], 
                                                                     updateProbe) # run the ptycho algorithm
        
        obj_recons[obj_pos[0]:obj_pos[1], obj_pos[2]:obj_pos[3]] = updated_local_obj # update only object ROI
        illum_recons = updated_local_illum # update only illumination ROI
        
    return obj_recons, illum_recons

def print_date_i():
    print('\n'+'EXECUTION BEGAN AT: ', end='')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\n')
    
def print_date_f():
    print('\n'+'EXECUTION FINISHED AT: ', end='')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\n')

def show_field_image(complex_array2D, ext=0):
    
    amp = np.abs(complex_array2D)
    phase = np.angle(complex_array2D)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches((12,4))
    if(ext != 0):        
        im1 = ax1.imshow(amp - np.min(amp), origin='lower', extent=np.array(ext)*1e6)
        im2 = ax2.imshow(phase - np.min(phase), origin='lower', extent=np.array(ext)*1e6)
        ax1.set_ylabel('y [$\mu m$]')
        ax1.set_xlabel('x [$\mu m$]')
        ax2.set_xlabel('x [$\mu m$]')
    else:
        im1 = ax1.imshow(amp - np.min(amp), origin='lower')
        im2 = ax2.imshow(phase - np.min(phase), origin='lower')
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    
    plt.show()
    
    
def save_output(output_filename, obj_recons, illum_recons, obj_grid, illum_grid, d={}):
    
    with h5py.File(output_filename, 'w') as f:
        
        ill_group = f.create_group('illumination')
        ill_amp = ill_group.create_dataset('amplitude', data=np.abs(illum_recons))
        ill_phase = ill_group.create_dataset('phase', data=np.angle(illum_recons))
        ill_group.attrs['mesh'] = illum_grid        


        obj_group = f.create_group('sample')
        obj_amp = obj_group.create_dataset('amplitude', data=np.abs(obj_recons))
        obj_phase = obj_group.create_dataset('phase', data=np.angle(obj_recons))
        obj_group.attrs['mesh'] = obj_grid

        # amp = np.abs(complex_array2D)
        # phase = np.angle(complex_array2D)


def normalize_matrix(matrix, n_max=1):
    
    if(n_max > 0):
        matrix_max = np.max(matrix)
        if(matrix_max > 0.0):
            matrix /= np.max(matrix)
            matrix *= n_max
        
        return matrix

def bin_matrix(matrix, binning_x, binning_y, plot_matrix=0):
    
    
    if((binning_x > 0) and (binning_y > 0)):
    
        xn = len(matrix[0,:])
        yn = len(matrix[:,0])
            
        if not((xn % binning_x == 0) & (yn % binning_y == 0)):
            print('array of shape ({0} x {1}) cannot be binned by factor {2} and {3}'.format(yn, xn, binning_y, binning_x))
        else:
            #print('binning diffraction pattterns by a factor y:{0} and x:{1}'.format(binning_y, binning_x))
            xn = int(xn / binning_x)
            yn = int(yn / binning_y)
    
            matrix_binned = np.zeros((yn,xn), dtype=float)
            
            count_y = 0
            for iy in range(yn):
    
                count_x = 0
                for ix in range(xn):
                    
                    matrix_binned[iy,ix] = np.sum(matrix[count_y:count_y+binning_y,
                                                           count_x:count_x+binning_x])
    
                    count_x += binning_x
                count_y += binning_y
            

            if(plot_matrix):
                fig, ax = plt.subplots(ncols=2)
                ax[0].imshow(np.log(matrix), origin='lower')
                ax[1].imshow(np.log(matrix_binned), origin='lower')
                plt.show()
            
            return matrix_binned

def crop_array(size, new_size):
    
    if(new_size < size):
    
        # print('cropping')
        # number of points is odd
        if (size % 2) != 0:
            mid_idx = int((size - 1) / 2)
        
        # number of points is even 
        else:
            mid_idx = int(size/2) # will get the pixel to the right as center
            
        # new n is odd
        if(new_size % 2) != 0:
            idx_min = int(mid_idx - (new_size-1)/2)
            idx_max = int(mid_idx + (new_size-1)/2 + 1)

        else:
            idx_min = int(mid_idx - new_size/2)            
            idx_max = int(mid_idx + new_size/2)
            
        new_index = [idx_min, idx_max]
                
        return new_index
        
    else:
        return [0, int(size)]
        
        

def crop_matrix(matrix, new_shape=[], plot_matrix=0):
    
    if(len(new_shape) == 2):
        
        crop_y, crop_x = new_shape
        yn, xn = matrix.shape
        yn_new, xn_new = new_shape
        
        new_idx_y = crop_array(yn, yn_new)
        new_idx_x = crop_array(xn, xn_new)
        
        matrix_cropped = matrix[new_idx_y[0] : new_idx_y[1],
                                new_idx_x[0] : new_idx_x[1]]   

        if(plot_matrix):
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(np.log10(matrix), origin='lower')
            ax[1].imshow(np.log10(matrix_cropped), origin='lower')
            plt.show()

        return matrix_cropped     
        
    else:
        return matrix
    

def add_noise_poisson(matrix, plot_matrix=0):
    
    noise = np.sqrt(np.random.poisson(matrix).astype('float'))
    matrix_noisy = matrix + noise
    
    if(plot_matrix):
        
        fig, ax = plt.subplots(figsize=(14,4), ncols=3)
        im0 = ax[0].imshow(np.log10(matrix), origin='lower')
        im1 = ax[1].imshow(np.log10(noise), origin='lower')    
        im2 = ax[2].imshow(np.log10(matrix_noisy), origin='lower')    
        fig.colorbar(im0, ax=ax[0])
        fig.colorbar(im1, ax=ax[1])
        fig.colorbar(im2, ax=ax[2])
    
    return matrix_noisy
    
def plot_diffraction_pattern(d, index):
    
    matrix = d['diff int'][index]
    
    fig, ax = plt.subplots(figsize=(10,4), ncols=2)
    im0 = ax[0].imshow(matrix, origin='lower')
    im1 = ax[1].imshow(np.log10(matrix), origin='lower')    
    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])
    
    

def loadDiffPatterns(filename, binning=[], cropping=[], 
                     add_noise=0, normalize_to=2**16, use_int=1):

    illum_positions = []
    obj_index = []
    diffInt = []
    
    with h5py.File(filename, 'r') as f:
        
        z = f.attrs['detector distance']        
        energy = f.attrs['energy']
        wl = 1.23984198e-6 / energy
        
        gname = 'diff wave'
        group = f[gname]
        dset_names = list(group.keys())
        n_dsets = len(dset_names)
        
        
        for i in range(n_dsets):
            
            dset = group[dset_names[i]]
            diffInt.append(np.array(dset))
            illum_positions.append(np.array([dset.attrs['x_transm'], dset.attrs['y_transm']]))
            
            if(i==0):
                
                xi = float(dset.attrs['xi'])*1e-3
                xf = float(dset.attrs['xf'])*1e-3
                xn = int(dset.attrs['xn'])
                yi = float(dset.attrs['yi'])*1e-3
                yf = float(dset.attrs['yf'])*1e-3
                yn = int(dset.attrs['yn'])
                
            
    
                
    if(cropping != []):
              
        if not (len(cropping) == 2):
            print('cropping must be a list of length 2, with ')
        else:
            print('cropping patterns to shape ({0},{1}) '.format(cropping[0], cropping[1]))            
            yn, xn = cropping
            
            diffIntAux = []           

            for pattern in diffInt:
                
                cropped_pattern = crop_matrix(pattern, cropping, 0)
                diffIntAux.append(cropped_pattern)
                                
            diffInt = diffIntAux
            del diffIntAux
                
    if(binning != []):

        if not((xn % binning[1] == 0) & (yn % binning[0] == 0)):
            print('array of shape ({0} x {1}) cannot be binned by factor {2} and {3}'.format(yn, xn, binning[0], binning[1]))
        else:
            print('binning diffraction pattterns by a factor y:{0} and x:{1}'.format(binning[0], binning[1]))
            xn = int(xn / binning[1])
            yn = int(yn / binning[0])
        
            diffIntAux = []
            
            for pattern in diffInt:
                
                diffIntAux.append(bin_matrix(pattern, binning[1], binning[0], 0))
                
            diffInt = diffIntAux
            del diffIntAux
            
    if(normalize_to > 0):
        
        print('normalizing to {0}'.format(normalize_to))

        diffIntAux = []
        
        for pattern in diffInt:
            
            diffIntAux.append(normalize_matrix(pattern, normalize_to))
            
        diffInt = diffIntAux
        del diffIntAux        
            
    if(add_noise):
        
        diffIntAux = []
            
        for pattern in diffInt:
            
            diffIntAux.append(add_noise_poisson(pattern, 1))
            
        diffInt = diffIntAux
        del diffIntAux
        
    if(use_int):
        
        diffIntAux = []
            
        for pattern in diffInt:
            
            diffIntAux.append(np.round(pattern).astype('int'))
            
        diffInt = diffIntAux
        
        
            
    
    rx = xf - xi                # range of the detector
    ry = yf - yi
    px = rx / (xn-1)            # pixel size of the detector
    py = ry / (yn-1)
    
    px_r = wl * z / (xn * px)   # reconstructed pixel size
    py_r = wl * z / (yn * py)
    rx_r = xn * px_r            # range of the beam reconstruction
    ry_r = yn * py_r                
     
    
    illum_positions = np.array(illum_positions)
    illum_shape = (yn, xn)
    ill_xi = -0.5 * illum_shape[1] * px_r 
    ill_xf = +0.5 * illum_shape[1] * px_r    
    ill_yi = -0.5 * illum_shape[0] * py_r 
    ill_yf = +0.5 * illum_shape[0] * py_r
    
    x_ill = np.linspace(ill_xi + 0.5*px_r, ill_xf - 0.5*px_r, illum_shape[1])
    y_ill = np.linspace(ill_yi + 0.5*py_r, ill_yf - 0.5*py_r, illum_shape[0])
    
    pos_max_x_abs = np.max(np.abs(illum_positions[:,0]))
    pos_max_y_abs = np.max(np.abs(illum_positions[:,1]))
    
    obj_shape_x  = illum_shape[1] + int(2 * np.ceil(pos_max_x_abs/px_r))    
    obj_shape_y  = illum_shape[0] + int(2 * np.ceil(pos_max_y_abs/py_r))
    
    obj_shape = (obj_shape_y, obj_shape_x)
    obj_xi = ill_xi - 0.5 * (obj_shape[1] - illum_shape[1]) * px_r 
    obj_xf = ill_xf + 0.5 * (obj_shape[1] - illum_shape[1]) * px_r
    obj_yi = ill_yi - 0.5 * (obj_shape[0] - illum_shape[0]) * py_r 
    obj_yf = ill_yf + 0.5 * (obj_shape[0] - illum_shape[0]) * py_r
    
    x_obj = np.linspace(obj_xi + 0.5*px_r, obj_xf - 0.5*px_r, obj_shape[1])
    y_obj = np.linspace(obj_yi + 0.5*py_r, obj_yf - 0.5*py_r, obj_shape[0])
    
    obj_positions = []
    for i in range(len(illum_positions)):
        xmin = np.abs( x_obj - (ill_xi-illum_positions[i][0]) ).argmin()
        ymin = np.abs( y_obj - (ill_yi-illum_positions[i][1]) ).argmin()
        obj_positions.append([ymin, ymin+illum_shape[0],
                              xmin, xmin+illum_shape[1]])
        
    
    d = dict()
    d['x_obj'] = x_obj
    d['y_obj'] = y_obj
    d['x_ill'] = x_ill
    d['y_ill'] = y_ill    
    d['diff int'] = diffInt
    d['positions'] = illum_positions
    d['obj_positions'] = obj_positions
    d['energy'] = energy
    d['wavelength'] = wl
    d['z'] = z
    
    return d































