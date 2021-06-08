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
from copy import deepcopy
import matplotlib.patches as patches

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
        plt.figure()
        plt.imshow(aperture.real)
        
    return aperture



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
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.tick_params(which='both', axis='both', top=1, right=1)
    ax2.tick_params(which='both', axis='both', top=1, right=1)
   
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


def normalize_matrix(matrix, n_max=1, marker=None):
    
    mask = matrix != marker
    
    matrix_max = np.max(matrix[mask])
    
    if(matrix_max > 0.0):
        matrix[mask] = matrix[mask] / matrix_max
        matrix[mask] = matrix[mask] * n_max
        
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

def crop_array(size, new_size, mid_idx=None):
    
    if(new_size < size):
    
        # print('cropping')
        # number of points is odd
        if(mid_idx is None):
            
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
        
        

def crop_matrix(matrix, new_idx_y, new_idx_x, plot_matrix=0):
    
        matrix_cropped = matrix[int(new_idx_y[0]) : int(new_idx_y[1]),
                                int(new_idx_x[0]) : int(new_idx_x[1])]   

        if(plot_matrix):
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(np.log10(matrix), origin='lower')
            ax[1].imshow(np.log10(matrix_cropped), origin='lower')
            plt.show()

        return matrix_cropped     
    

def add_noise_poisson(matrix, marker=None, plot_matrix=0):

    mask = matrix != marker
    noisy = matrix
    
    matrix_aux = matrix
    matrix_aux[matrix_aux < 0] = 0
    noise = np.random.poisson(matrix_aux).astype('float')
    noise[noise < 0] = 0
    
    noisy[mask] = noise[mask]               
    
    if(plot_matrix):
        
        fig, ax = plt.subplots(figsize=(14,4), ncols=2)
        im0 = ax[0].imshow(np.log10(matrix), origin='lower')
        im1 = ax[1].imshow(np.log10(noisy), origin='lower')    
        fig.colorbar(im0, ax=ax[0])
        fig.colorbar(im1, ax=ax[1])
    
    return noisy
    
def plot_diffraction_pattern(d, index):
    
    matrix = d['diff int'][index]
    
    fig, ax = plt.subplots(figsize=(10,4), ncols=2)
    im0 = ax[0].imshow(matrix, origin='lower')
    im1 = ax[1].imshow(np.log10(matrix), origin='lower')    
    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])
    
    
def create_circular_beamstop(matrix, center=None, radius=None, marker=-1):

    h, w = matrix.shape    

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    
    matrix[mask] = marker
    
    return matrix
    
def mark_bad_pixels(matrix, bad_pixels_list, marker=-1):
    for px in bad_pixels_list:
        matrix[px[0], px[1]] = marker
    return matrix

def mark_bad_columns(matrix, bad_columns_list, marker=-1):
    for px in bad_columns_list:
        matrix[:, px] = marker
    return matrix

def mark_bad_lines(matrix, bad_lines_list, marker=-1):
    for px in bad_lines_list:
        matrix[px, :] = marker
    return matrix
    
    
def read_nexus_file(filename, detectors_list, motors_list, 
                    group_path='Scan/scan_000/instrument'):
    
    energy_path = 'pre_scan/beamline_status/Monochromator/energy_error'
    detector_distance_path = ''
    
    with h5py.File(filename, 'r') as f:
        
        group = f[group_path]
        
        # store selected detectors in this array
        detector_array = []
        for detector in detectors_list:
            detector_array.append(np.array(group[detector + '/data'][()]))
        
        # store selected motors in this array    
        motor_array = []
        for motor in motors_list:
            motor_array.append(np.array(group[motor+'/data'][()]))
            
        energy = np.array(f[energy_path][()])
        if(detector_distance_path == ''):
            detector_distance = 1.0
        else:
            detector_distance = np.array(f[detector_distance_path][()])
            
        attributes = {'energy':energy, 'detector_distance':detector_distance}
            
    return detector_array, motor_array, attributes
    
def pre_process_from_nexus(filename, detectors, motors, 
                           cropping=[], center_pixel=[],
                           bad_pixels=[], binning=[]):
    
    detectors_list, motors_list = read_nexus_file(filename, detectors, motors)
    
    # x = motor1[0,:][:-40]
    # z = motor2[:-1,0]
    
    return 0 
    

def updateIllum(obj, illum, exitWave, exitWave_forcedAmp):
    beta = 0.1
    illum_updated = illum + beta * ( np.conjugate(obj) / (np.max(np.abs(obj))**2 + 1e-5 ) ) * (exitWave_forcedAmp - exitWave) 
    return illum_updated
    
def updateObj(obj, illum, exitWave, exitWave_forcedAmp):
    alpha = 0.9
    obj_updated = obj + alpha * ( np.conjugate(illum) / (np.max(np.abs(illum))**2 + 1e-5) ) * (exitWave_forcedAmp - exitWave) 
    return obj_updated

def run_local_iteration(obj, illum, diffPattern, updateProbe=1, updateSample=1, marker=None):
    
    # print('obj shape = ', obj.shape)
    # print('illum shape = ', illum.shape)
    
    exitWave = obj * illum # create exit wave
    F_exitWave = np.fft.fft2(exitWave) # fourier transform

    diffPattern = np.fft.fftshift(diffPattern)
    
    # create mask to replace only the known pixels
    mask = diffPattern != marker
        
    # replace modulus
    F_exitWave_abs = np.abs(F_exitWave) # get exit waves amplitudes
    F_exitWave_abs[F_exitWave_abs<=0] = 1.0 # avoid division by zero
    F_exitWave = F_exitWave / F_exitWave_abs # normalize the amplitudes
        
    # print('mask shape = ', mask.shape)
    # print('F_exitWave shape = ', F_exitWave.shape)
    # print('diffPattern shape = ', diffPattern.shape)
    
    F_exitWave[mask] = F_exitWave[mask] * np.sqrt(diffPattern[mask]) # replace modulus by diffraction intensity
    
    exitWave_forcedAmp = np.fft.ifft2(F_exitWave) # inverse transform
 
    if(updateSample):
        obj = updateObj(obj, illum, exitWave, exitWave_forcedAmp) # update object
    
    if(updateProbe):
        illum = updateIllum(obj, illum, exitWave, exitWave_forcedAmp) # update probe 
    
    return obj, illum

def run_ePIE_interation(diffPatterns, obj_recons, illum_recons, obj_positions, 
                        updateProbe=1, updateSample=1, marker=None):
    
    positionsIndex = np.random.permutation(len(diffPatterns)) # select positions randomly
    # print(positionsIndex)
    
    for i in positionsIndex: # random order
                
        obj_pos = obj_positions[i] # object positions for this step
        
        local_obj = obj_recons[obj_pos[0]:obj_pos[1], obj_pos[2]:obj_pos[3]] # select object ROI
        local_illum = illum_recons # select illumination ROI
        
        updated_local_obj, updated_local_illum = run_local_iteration(local_obj, 
                                                                     local_illum, 
                                                                     diffPatterns[i], 
                                                                     updateProbe,
                                                                     updateSample,
                                                                     marker) # run the ptycho algorithm
        
        obj_recons[obj_pos[0]:obj_pos[1], obj_pos[2]:obj_pos[3]] = updated_local_obj # update only object ROI
        illum_recons = updated_local_illum # update only illumination ROI
        
    return obj_recons, illum_recons

def loadDiffPatterns(filename, binning=[], cropping=[], multi_factor=0,
                     add_noise=0, normalize_to=0, ceil=0, use_int=0, around=0,
                     beamstop=0, marker=-1, specific_datasets=[], old=0):

   
    with h5py.File(filename, 'r') as f:
        
        z = f.attrs['detector distance']        
        energy = f.attrs['energy']
        wl = 1.23984198e-6 / energy
        
        # gname = 'diffracted intensity'
        gname = 'diff wave'
        group = f[gname]
        
        if(specific_datasets != []):
            dset_names = specific_datasets                    
        else:
            dset_names = list(group.keys())
        
        n_dsets = len(dset_names)
        print('n datasets =', n_dsets)
        
        if(old):

            dset = group[dset_names[0]]
            xi = float(dset.attrs['xi'])*1e-3
            xf = float(dset.attrs['xf'])*1e-3
            xn = int(dset.attrs['xn'])
            yi = float(dset.attrs['yi'])*1e-3
            yf = float(dset.attrs['yf'])*1e-3
            yn = int(dset.attrs['yn'])
    
        else:
            xi = float(group.attrs['xi'])
            xf = float(group.attrs['xf'])
            xn = int(group.attrs['xn'])
            yi = float(group.attrs['yi'])
            yf = float(group.attrs['yf'])
            yn = int(group.attrs['yn']) 
        
        print("nx, ny =", xn, yn)
        xn0 = deepcopy(xn)
        yn0 = deepcopy(yn)
        
        illum_positions = []
        diffInt = []
        max_count = 0
        
        for i in range(n_dsets):
            
            dset = group[dset_names[i]]
            pattern = np.array(dset)
            
            if(old):
                illum_positions.append(np.array([dset.attrs['x_transm'], dset.attrs['y_transm']]))
            else:
                illum_positions.append(np.array([dset.attrs['x_sample'], dset.attrs['y_sample']]))

            xn = xn0
            yn = yn0

            if(cropping != []):
              
                if not (len(cropping) == 2):
                    if(i==0): print('cropping must be a list of length 2, with ')
                else:
                    if(i==0): 
                        print('cropping patterns to shape ({0},{1}) '.format(cropping[0], cropping[1]))                
                        
                        x_array = np.linspace(xi, xf, xn)
                        y_array = np.linspace(yi, yf, yn)
                        x_idx_crop = crop_array(xn, cropping[1])                        
                        y_idx_crop = crop_array(yn, cropping[0])
                        
                        # update initial and final values
                        xi = x_array[x_idx_crop[0]]
                        xf = x_array[x_idx_crop[1]]
                        yi = y_array[y_idx_crop[0]]
                        yf = y_array[y_idx_crop[1]]
                        
                    pattern = crop_matrix(pattern, y_idx_crop, x_idx_crop, plot_matrix=0)
                    
                    # update number of points
                    yn, xn = cropping                    
              
            if(binning != []):
        
                if not((xn % binning[1] == 0) & (yn % binning[0] == 0)):
                    if(i==0): print('array of shape ({0} x {1}) cannot be binned by factor {2} and {3}'.format(yn, xn, binning[0], binning[1]))
                else:
                    if(i==0): print('binning diffraction pattterns by a factor y:{0} and x:{1}'.format(binning[0], binning[1]))
                    xn = int(xn / binning[1])
                    yn = int(yn / binning[0])
                    pattern = bin_matrix(pattern, binning[1], binning[0], 0)

            if(beamstop != 0):
                
                if(i==0): print('adding beamstop')
                pattern = create_circular_beamstop(pattern, radius=beamstop, marker=marker)
                    
            if(normalize_to > 0):
        
                if(i==0): print('normalizing to {0}'.format(normalize_to))
                pattern = normalize_matrix(pattern, normalize_to, marker=marker)

            if(multi_factor > 0):
        
                if(i==0): print('using multiplication factor = {0}'.format(multi_factor))
                pattern = pattern * multi_factor
                                                       
            if(add_noise):
                
                if(i==0): print('adding poisson noise')
                pattern = add_noise_poisson(pattern, marker=marker, plot_matrix=0)
                
                # normalize again
                if(normalize_to > 0):
                    pattern = normalize_matrix(pattern, normalize_to, marker=marker)
                
            if(ceil):
                pattern[pattern != marker] = np.ceil(pattern[pattern != marker])
                
            if(around):
                pattern[pattern != marker] = np.around(pattern[pattern != marker])
                
            if(use_int):
                if(i==0): print('converting to int')
                pattern[pattern != marker] = np.round(pattern[pattern != marker]).astype('int')

            max_c = np.max(pattern)
            if(max_c > max_count):
                max_count = max_c
               
            diffInt.append(pattern)
    
    print("maximum count = {0:.4e}".format(max_count))
    
    rx = xf - xi                # range of the detector
    ry = yf - yi
    px = rx / (xn-1)            # pixel size of the detector
    py = ry / (yn-1)
    
    px_r = wl * z / (xn * px)   # reconstructed pixel size
    py_r = wl * z / (yn * py)
    #rx_r = xn * px_r            # range of the beam reconstruction
    #ry_r = yn * py_r                
     
    
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
    d['max count'] = max_count
    
    return d


def run_ptycho_reconstruction(n_iterations, obj_guess, illum_guess, 
                              diff_patterns, obj_positions, obj_grid, illum_grid, 
                              output_filename, fixed_probe=0, marker=-1):

    startTime = time.time()
    
    obj_recons = obj_guess
    illum_recons = illum_guess

    n_patterns = len(diff_patterns)
    
    # n_iterations = 4000
    print('Starting reconstruction algorithm with {0:d} iterations'.format(n_iterations))
    progress = np.linspace(0, n_iterations, 21)
    count=1
    for i in range(n_iterations):
        updateProbe = 0 if i < n_iterations * fixed_probe else 1
        # updateProbe = 0 
        
        obj_recons, illum_recons = run_ePIE_interation(diff_patterns, obj_recons, illum_recons, obj_positions, 
                                                       updateProbe, marker=marker)
        if(i >= progress[count]):
            print("finished iteration {0} out of {1}".format(i, n_iterations))
            show_field_image(obj_recons, ext=obj_grid)
            show_field_image(illum_recons, ext=illum_grid)
            count += 1 

    print("N PATTERNS = ", n_patterns)
    # print("OVERLAPPING ", overlapping)
    print("N ITERATIONS = {0}".format(n_iterations))
    print("ELAPSED TIME = {0:.1f} minutes".format((time.time() - startTime)/60.0))

    show_field_image(obj_recons, ext=obj_grid)
    show_field_image(illum_recons, ext=illum_grid)
    
    save_output(output_filename, obj_recons, illum_recons, obj_grid, illum_grid)

    return obj_recons, illum_recons


def plot_reconstruction_from_hdf5_joined(h5_filename, sample_lim, beam_lim,
                                         sample_bar_nm=50, beam_bar_nm=100, 
                                         show_ab=0, posfix='', crop=0):
        
    sample_cmap = 'viridis'
    beam_cmap = 'viridis'
    
    
    with h5py.File(h5_filename, 'r') as f:
        
        beam_amp = np.array(f['illumination']['amplitude'])
        beam_phase = np.array(f['illumination']['phase'])
        beam_grid = np.array(f['illumination'].attrs['mesh'])*1e6
    
        sample_amp = np.array(f['sample']['amplitude'])
        sample_phase = np.array(f['sample']['phase']) 
        sample_grid = np.array(f['sample'].attrs['mesh'])*1e6
    
    
    if(crop):
        beam_ny, beam_nx = beam_amp.shape
        beam_x = np.linspace(beam_grid[0], beam_grid[1], beam_nx)
        beam_y = np.linspace(beam_grid[2], beam_grid[3], beam_ny)
        
        beam_idx_x = [np.abs(beam_x - beam_lim[0]).argmin(),
                      np.abs(beam_x - beam_lim[1]).argmin()]

        beam_idx_y = [np.abs(beam_y - beam_lim[0]).argmin(),
                      np.abs(beam_y - beam_lim[1]).argmin()]

        beam_amp = crop_matrix(beam_amp, beam_idx_y, beam_idx_x)
        beam_phase = crop_matrix(beam_phase, beam_idx_y, beam_idx_x)

        sample_ny, sample_nx = sample_amp.shape
        sample_x = np.linspace(sample_grid[0], sample_grid[1], sample_nx)
        sample_y = np.linspace(sample_grid[2], sample_grid[3], sample_ny)
        
        sample_idx_x = [np.abs(sample_x - sample_lim[0]).argmin(),
                      np.abs(sample_x - sample_lim[1]).argmin()]

        sample_idx_y = [np.abs(sample_y - sample_lim[0]).argmin(),
                      np.abs(sample_y - sample_lim[1]).argmin()]
        
        sample_amp = crop_matrix(sample_amp, sample_idx_y, sample_idx_x)
        sample_phase = crop_matrix(sample_phase, sample_idx_y, sample_idx_x)

        beam_grid = [beam_x[beam_idx_x[0]], beam_x[beam_idx_x[1]], 
                     beam_y[beam_idx_y[0]], beam_y[beam_idx_x[1]]]
        
        sample_grid = [sample_x[sample_idx_x[0]], sample_x[sample_idx_x[1]], 
                     sample_y[sample_idx_y[0]], sample_y[sample_idx_x[1]]]
        
    beam_amp = beam_amp / np.max(beam_amp)
    sample_amp = sample_amp / np.max(sample_amp)
    sample_phase = sample_phase - np.mean(sample_phase)
    
    #########################################################################
    ## plot beam
    #########################################################################
    
    
    fig, ax = plt.subplots(figsize=(5.0,3), nrows=1, ncols=2)
    fig.tight_layout()
    fig.subplots_adjust(0.01, 0.1, 0.99, 0.99, wspace=0.02)
    # fig.subplots_adjust(wspace=0.01)
    
    im0 = ax[0].imshow(beam_amp, origin='lower', cmap=beam_cmap, 
                       extent=beam_grid, aspect='equal', vmin=0, vmax=1)
    
    # cax0 = make_axes_locatable(ax[0]).append_axes('bottom', size="5%", pad=0.05) 
    cb0 = fig.colorbar(im0, ax=ax[0], orientation='horizontal',
                       pad=0.04, fraction=0.04, ticks=[0,0.5,1])
    cb0.ax.minorticks_on()
    
    im1 = ax[1].imshow(beam_phase, origin='lower', cmap='hsv',
                       vmin=-np.pi, vmax=np.pi, extent=beam_grid, aspect='equal')
    
    # cax1 = make_axes_locatable(ax[1]).append_axes('bottom', size="5%", pad=0.05) 
    cb1 = fig.colorbar(im1, ax=ax[1], orientation='horizontal',
                       pad=0.04, fraction=0.04, 
                       ticks=[-np.pi,0,np.pi])
    # cb1.ax.set_xticklabels(['-\u03C0', 0, '+\u03C0'])
    cb1.ax.set_xticklabels(['$-\pi$', 0, '$+\pi$'])
    cb1.ax.minorticks_on()
    
    if(beam_lim != []):
        ax[0].set_xlim(beam_lim[0], beam_lim[1])
        ax[1].set_xlim(beam_lim[0], beam_lim[1])
        ax[0].set_ylim(beam_lim[0], beam_lim[1])
        ax[1].set_ylim(beam_lim[0], beam_lim[1])
        x_range = (beam_lim[1] - beam_lim[0])*1e3
    
    ### add scale bar
    if(beam_lim == []):
        x_range = (beam_grid[1] - beam_grid[0])*1e3 # nm
    bar_width = beam_bar_nm # nm
    bar_relative = bar_width / x_range
    
    rect = patches.Rectangle((1 - bar_relative - 0.02, 0.02), bar_relative, bar_relative/5, linewidth=0, 
                             edgecolor='w', facecolor='w', transform=ax[0].transAxes)
    ax[0].add_patch(rect)
    ax[0].text(0.98, 0.04 + bar_relative/5, '{0} nm'.format(beam_bar_nm), color='w', weight='bold',
               fontsize=7.5, horizontalalignment='right', transform=ax[0].transAxes)
    
    if(show_ab):
        ax[0].text(0.04, 0.90, '(a)', color='w', weight='bold',
                   fontsize=12, transform=ax[0].transAxes)
        ax[1].text(0.04, 0.90, '(b)', color='w', weight='bold', backgroundcolor='darkblue',
                   fontsize=12, transform=ax[1].transAxes)
    
    
    ax[0].axis('off')
    ax[1].axis('off')


    plt.savefig(h5_filename[:-3] + posfix + '_probe.png', dpi=400)
    
    #########################################################################
    ## plot sample
    #########################################################################
    
    
    fig, ax = plt.subplots(figsize=(5.0,3), nrows=1, ncols=2)
    fig.tight_layout()
    fig.subplots_adjust(0.01, 0.1, 0.99, 0.99, wspace=0.02)
    # fig.subplots_adjust(wspace=0.01)
    
    im0 = ax[0].imshow(sample_amp, origin='lower', cmap=sample_cmap, 
                       extent=sample_grid, aspect='equal', vmin=0, vmax=1)
    
    # cax0 = make_axes_locatable(ax[0]).append_axes('bottom', size="5%", pad=0.05) 
    cb0 = fig.colorbar(im0, ax=ax[0], orientation='horizontal',
                       pad=0.04, fraction=0.04, ticks=[0,0.5,1])
    cb0.ax.minorticks_on()
    
    im1 = ax[1].imshow(sample_phase, origin='lower', cmap='hsv',
                       vmin=-np.pi, vmax=np.pi, extent=sample_grid, aspect='equal')
    
    # cax1 = make_axes_locatable(ax[1]).append_axes('bottom', size="5%", pad=0.05) 
    cb1 = fig.colorbar(im1, ax=ax[1], orientation='horizontal',
                       pad=0.04, fraction=0.04, 
                       ticks=[-np.pi,0,np.pi])
    # cb1.ax.set_xticklabels(['-\u03C0', 0, '+\u03C0'])
    cb1.ax.set_xticklabels(['$-\pi$', 0, '$+\pi$'])
    cb1.ax.minorticks_on()
    
    if(sample_lim != []):
        ax[0].set_xlim(sample_lim[0], sample_lim[1])
        ax[1].set_xlim(sample_lim[0], sample_lim[1])
        ax[0].set_ylim(sample_lim[0], sample_lim[1])
        ax[1].set_ylim(sample_lim[0], sample_lim[1])
        x_range = (sample_lim[1] - sample_lim[0])*1e3
    
    ### add scale bar
    if(sample_lim == []):
        x_range = (sample_grid[1] - sample_grid[0])*1e3 # nm
    bar_width = sample_bar_nm # nm
    bar_relative = bar_width / x_range
    
    rect = patches.Rectangle((1 - bar_relative - 0.02, 0.02), bar_relative, bar_relative/5, linewidth=0, 
                             edgecolor='w', facecolor='w', transform=ax[0].transAxes)
    ax[0].add_patch(rect)
    ax[0].text(0.98, 0.04 + bar_relative/5, '{0} nm'.format(sample_bar_nm), color='w', weight='bold',
               fontsize=7.5, horizontalalignment='right', transform=ax[0].transAxes)
    
    if(show_ab):
        ax[0].text(0.04, 0.90, '(a)', color='w', weight='bold',
                   fontsize=12, transform=ax[0].transAxes)
        ax[1].text(0.04, 0.90, '(b)', color='w', weight='bold', backgroundcolor='darkblue',
                   fontsize=12, transform=ax[1].transAxes)
    
    
    # cb1.set_tick_params(direction='in')
    
    ax[0].axis('off')
    ax[1].axis('off')
       
    
    plt.savefig(h5_filename[:-3] + posfix + '_sample.png', dpi=400)




























