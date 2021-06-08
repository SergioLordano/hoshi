#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:44:18 2018

@author: sergio.lordano
"""
def get_fwhm(x, y, oversampling=1, zero_padding=False, avg_correction=False, debug=False):
    
    import numpy as np
    from scipy.interpolate import interp1d
    
    def add_zeros(array):
        aux = []
        aux.append(0)
        for i in range(len(array)):
            aux.append(array[i])
        aux.append(0)
        return np.array(aux)
    
    def add_steps(array):
        aux = []
        step = (np.max(array)-np.min(array))/(len(array)-1)
        aux.append(array[0]-step)
        for i in range(len(array)):
            aux.append(array[i])
        aux.append(array[-1]+step)
        return np.array(aux)
    
    def interp_distribution(array_x,array_y,oversampling):
        dist = interp1d(array_x, array_y)
        x_int = np.linspace(np.min(array_x), np.max(array_x), int(len(x)*oversampling))
        y_int = dist(x_int)
        return x_int, y_int 
    
    if(oversampling > 1.0):
        array_x, array_y = interp_distribution(x, y, oversampling)
    else:
        array_x, array_y = x, y
        
    if(zero_padding):
        array_x = add_steps(x)
        array_y = add_zeros(y)
        
    try:    
        y_peak = np.max(array_y)
        idx_peak = (np.abs(array_y-y_peak)).argmin()
        #x_peak = array_x[idx_peak]
        if(idx_peak==0):
            left_hwhm_idx = 0
        else:
            #left_hwhm_idx = (np.abs(array_y[:idx_peak]-y_peak/2)).argmin()
#            for i in range(idx_peak,0,-1):
            for i in range(0,idx_peak):
#                if np.abs(array_y[i]-y_peak/2)<np.abs(array_y[i-1]-y_peak/2) and (array_y[i-1]-y_peak/2)<0:
                if np.abs(array_y[i]-y_peak/2)>np.abs(array_y[i-1]-y_peak/2) and (array_y[i-1]-y_peak/2)>0:
                    break                
            left_hwhm_idx = i     
            
        if(idx_peak==len(array_y)-1):
            right_hwhm_idx = len(array_y)-1
        else:
            #right_hwhm_idx = (np.abs(array_y[idx_peak:]-y_peak/2)).argmin() + idx_peak
#            for j in range(idx_peak,len(array_y)):
            for j in range(len(array_y)-2, idx_peak, -1):
#                if np.abs(array_y[j]-y_peak/2)<np.abs(array_y[j+1]-y_peak/2) and (array_y[j+1]-y_peak/2)<0:
                if np.abs(array_y[j]-y_peak/2)>np.abs(array_y[j+1]-y_peak/2) and (array_y[j+1]-y_peak/2)>0:
                    break              
            right_hwhm_idx = j
            
        fwhm = array_x[right_hwhm_idx] - array_x[left_hwhm_idx]               
            
        if(avg_correction):
            avg_y = (array_y[left_hwhm_idx]+array_y[right_hwhm_idx])/2.0
            popt_left = np.polyfit(np.array([array_x[left_hwhm_idx-1],array_x[left_hwhm_idx],array_x[left_hwhm_idx+1]]),
                                   np.array([array_y[left_hwhm_idx-1],array_y[left_hwhm_idx],array_y[left_hwhm_idx+1]]),1)                                   
            popt_right = np.polyfit(np.array([array_x[right_hwhm_idx-1],array_x[right_hwhm_idx],array_x[right_hwhm_idx+1]]),
                                   np.array([array_y[right_hwhm_idx-1],array_y[right_hwhm_idx],array_y[right_hwhm_idx+1]]),1)
            x_left = (avg_y-popt_left[1])/popt_left[0]
            x_right = (avg_y-popt_right[1])/popt_right[0]
            fwhm = x_right - x_left
            
            return [fwhm, x_left, x_right, avg_y, avg_y]
        else:
            return [fwhm, array_x[left_hwhm_idx], array_x[right_hwhm_idx], array_y[left_hwhm_idx], array_y[right_hwhm_idx]]
            
        if(debug):
            print(y_peak)
            print(idx_peak)
            print(left_hwhm_idx, right_hwhm_idx)
            print(array_x[left_hwhm_idx], array_x[right_hwhm_idx])            
        
    except ValueError:
        fwhm = 0.0        
        print("Could not calculate fwhm\n")   
        return [fwhm, 0, 0, 0, 0]
    
    
def plot_caustic2D_cut(filename_h5='test.h5', show_axis='x', fixed_position=0.0, unitFactor=[1.0, 1e6], unitLabel=['Z [m]', 'X [$\mu m$]'], aspect='auto', xlim=None, ylim=None, zlim=None, scale='linear', figsize=(8,6), fontsize=12):
    
    import h5py
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    
    f = h5py.File(filename_h5, 'r')
    
    xmin = f.attrs['xStart']
    xmax = f.attrs['xFin']
    nx = f.attrs['nx']
    ymin = f.attrs['yStart']
    ymax = f.attrs['yFin']
    ny = f.attrs['ny']
    emin = f.attrs['eStart']
    emax = f.attrs['eFin']
    ne = f.attrs['ne']
    zOffset = f.attrs['zOffset']
    zmin = f.attrs['zStart'] + zOffset
    zmax = f.attrs['zFin'] + zOffset
    nz = f.attrs['nz']
    zStep = f.attrs['zStep']
    
    
    mesh = [xmin, xmax, nx, ymin, ymax, ny, emin, emax, ne, zmin, zmax, nz]
    uz_list = []
    
    plt.figure(figsize=figsize)
    plt.subplots_adjust(0.1,0.18,0.99,0.95)
    
    plt.ylabel(unitLabel[1], fontsize=fontsize)
    plt.xlabel(unitLabel[0], fontsize=fontsize)
    
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    
    if(show_axis == 'x'):
        
        posfix = '_caustic_x'
        
        y_pts = np.linspace(ymin, ymax, ny)
        fixed_idx = np.abs(fixed_position - y_pts).argmin()
    
        for i in range(nz):
            dataset = 'step_{0}'.format(i)
            uz_list.append(f[dataset][fixed_idx, :])
            uz = np.array(uz_list).transpose()
            
            if(zlim is not None):
                uz_min, uz_max = zlim[0], zlim[1]
            else:
                uz_min, uz_max = np.min(uz), np.max(uz)
                
            if(scale=='linear'):
                plt.imshow(uz, extent=[zmin*unitFactor[0], zmax*unitFactor[0], xmin*unitFactor[1], xmax*unitFactor[1]], aspect=aspect, origin='lower', vmin=uz_min, vmax=uz_max)
            
            elif(scale=='log'):
                if(np.min(uz) <= 0.0):
                    uz_min_except_0 = np.min(uz[uz>0])
                    uz[uz<=0.0] = uz_min_except_0/2.0
                    uz_min = uz_min_except_0/2.0
                    plt.imshow(uz, extent=[zmin*unitFactor[0], zmax*unitFactor[0], xmin*unitFactor[1], xmax*unitFactor[1]], aspect=aspect, origin='lower', norm=LogNorm(vmin=uz_min, vmax=uz_max))
                else:
                    plt.imshow(uz, extent=[zmin*unitFactor[0], zmax*unitFactor[0], xmin*unitFactor[1], xmax*unitFactor[1]], aspect=aspect, origin='lower', norm=LogNorm(vmin=uz_min, vmax=uz_max))
               
    if(show_axis == 'y'):
        
        posfix = '_caustic_y'        
        
        x_pts = np.linspace(xmin, xmax, nx)
        fixed_idx = np.abs(fixed_position - x_pts).argmin()
    
        for i in range(nz):
            dataset = 'step_{0}'.format(i)
            uz_list.append(f[dataset][:, fixed_idx])
            uz = np.array(uz_list).transpose()
            
            if(zlim is not None):
                uz_min, uz_max = zlim[0], zlim[1]
            else:
                uz_min, uz_max = np.min(uz), np.max(uz)
                
            if(scale=='linear'):
                plt.imshow(uz, extent=[zmin*unitFactor[0], zmax*unitFactor[0], ymin*unitFactor[1], ymax*unitFactor[1]], aspect=aspect, origin='lower', vmin=uz_min, vmax=uz_max)
            
            elif(scale=='log'):
                if(np.min(uz) <= 0.0):
                    uz_min_except_0 = np.min(uz[uz>0])
                    uz[uz<=0.0] = uz_min_except_0/2.0
                    uz_min = uz_min_except_0/2.0                    
                    plt.imshow(uz, extent=[zmin*unitFactor[0], zmax*unitFactor[0], ymin*unitFactor[1], ymax*unitFactor[1]], aspect=aspect, origin='lower', norm=LogNorm(vmin=uz_min, vmax=uz_max))
                else:
                    plt.imshow(uz, extent=[zmin*unitFactor[0], zmax*unitFactor[0], ymin*unitFactor[1], ymax*unitFactor[1]], aspect=aspect, origin='lower', norm=LogNorm(vmin=uz_min, vmax=uz_max))
    ax = plt.gca()
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=fontsize)
    plt.minorticks_on()
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        
    f.close()
    
    plt.savefig(filename_h5[:-3]+posfix+'.png', dpi=400)    
    
    plt.show()
    
    return mesh, uz




def plot_caustic_slice(filename='test.h5', z=0.0, dataset_prefix='step_', plot='cut',
                       unitFactor=1000.0, xlabel='X', ylabel='Z', units='[$\mu m$]', 
                       xlim=None, ylim=None, scale='linear', showPlot=True, showFWHM=False):
    
    import h5py
    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # ==================================================================== #
    # === READING DATA =================================================== #
    # ==================================================================== #
    
    f = h5py.File(filename, 'r')
    
    xmin = f.attrs['xStart']*unitFactor
    xmax = f.attrs['xFin']*unitFactor
    nx = f.attrs['nx']
    ymin = f.attrs['yStart']*unitFactor
    ymax = f.attrs['yFin']*unitFactor
    ny = f.attrs['ny']
    zOffset = f.attrs['zOffset'] 
    zmin = f.attrs['zStart'] + zOffset
    zmax = f.attrs['zFin'] + zOffset
    nz = f.attrs['nz']
    zStep = f.attrs['zStep']
    max_int = f.attrs['max peak int']
    max_int_z = f.attrs['max peak z']
    
    x_axis = np.linspace(xmin, xmax, nx)
    y_axis = np.linspace(ymin, ymax, ny)
    z_axis = np.linspace(zmin, zmax, nz)
    
    z_idx = np.abs(z_axis - z).argmin()
    dset_name = dataset_prefix + str(z_idx)
    dset_z = f[dset_name].attrs['z']
    dset_max_int = f[dset_name].attrs['max intensity']
    dset_integral = f[dset_name].attrs['integral']
    xz = np.array(f[dset_name])
        
    f.close()
    
    x_cut = xz[np.abs(y_axis).argmin(), :]
    y_cut = xz[:, np.abs(x_axis).argmin()]
    
    
    # ==================================================================== #
    # === PLOTTING DATA ================================================== #
    # ==================================================================== #
    
    fig, ax2D = plt.subplots(figsize=(10.0, 6.5))
    fontsize=12       

    # Creates Dependent Axes
    divider = make_axes_locatable(ax2D)
    axX = divider.append_axes("top", 1.6, pad=0.1, sharex=ax2D)
    axY = divider.append_axes("left", 1.6, pad=0.1, sharey=ax2D)
    axT = divider.append_axes("right", 2.5, pad=1.0, sharey=ax2D)
    axY.invert_xaxis()

    # Adjust positions
    pos2D = ax2D.get_position()        
    ax2D.set_position([pos2D.x0-0.115, pos2D.y0-0.03 , pos2D.width+0.20, pos2D.height+0.12])

    # Adjust Ticks
    axX.xaxis.set_major_locator(plt.MaxNLocator(5))
    axX.minorticks_on()
    axY.minorticks_on()
    ax2D.minorticks_on()
        
    axX.xaxis.set_tick_params(which='both', direction='in', top=True, bottom=True, labelbottom=False)
    axX.yaxis.set_tick_params(which='both', direction='in', left=True, right=True)
    axY.yaxis.set_tick_params(which='both', direction='in', left=True, right=True, labelleft=False, labelright=False)
    axY.xaxis.set_tick_params(which='both', direction='in', top=True, bottom=True)
    ax2D.yaxis.set_label_position("right")
    ax2D.tick_params(axis='both', which='both', direction='in', left=True,top=True,right=True,bottom=True,labelleft=False,labeltop=False, labelright=True,labelbottom=True, labelsize=fontsize)
    axT.tick_params(axis='both',which='both',left=False,top=False,right=False,bottom=False,labelleft=False,labeltop=False, labelright=False,labelbottom=False)

       
    # Write Labels
    ax2D.set_xlabel(xlabel + ' ' + units, fontsize=fontsize)
    ax2D.set_ylabel(ylabel + ' ' + units, fontsize=fontsize)
    
    # Plots data    
    if(scale=='linear'):
        ax2D.imshow(xz, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower') # 2D data

    if(plot=='cut'):
        
        axX.plot(x_axis, x_cut, '-C0')
        axY.plot(y_cut, y_axis, '-C0')


    # Defines limits
    if(xlim is not None):
        ax2D.set_xlim(xlim[0], xlim[1])

    if(ylim is not None):
        ax2D.set_ylim(ylim[0], ylim[1])
    
    if(showFWHM):
        fwhmx = get_fwhm(x_axis, x_cut, oversampling=30, zero_padding=False, avg_correction=False, debug=False)
        fwhmy = get_fwhm(y_axis, y_cut, oversampling=30, zero_padding=False, avg_correction=False, debug=False)
    
    text0  = 'FILE: \n' 
    text0 += filename + '\n\n'
    text0 += 'DATASET: \n'
    text0 += dset_name + '\n\n'
    text0 += 'z = {0:.3f} m\n'.format(dset_z)
    text0 += 'max. intensity = {0:.2e} \n'.format(dset_max_int)
    text0 += 'ph/s/mm$^2$/0.1%bw' + '\n'
    text0 += 'integral = {0:.2e} \n'.format(dset_integral)
    text0 += 'ph/s/0.1%bw' + '\n\n'
    
    if(showFWHM):
        text0 += 'X FWHM = {0:.3f}\n'.format(fwhmx[0])
        text0 += 'Z FWHM = {0:.3f}\n\n'.format(fwhmy[0])
    
    text0 += 'GROUP: \n\n'
    text0 += 'max. intensity = {0:.2e} \n'.format(max_int)
    text0 += 'at z = {0:.3f} m\n'.format(max_int_z)
    
    axT.text(0.03, 0.97, text0, color='black', family='serif', weight='medium', 
             horizontalalignment='left', verticalalignment='top', 
             fontsize=11, transform= axT.transAxes)
    
    plt.show()



def plot_caustic_slice_int(**kwargs):
    
    import ipywidgets as wg
    from IPython.display import display  
    
    zvalue = wg.FloatText(value=kwargs['init_value'])
    zslider = wg.FloatSlider(value=kwargs['init_value'], min=kwargs['min_value'], max=kwargs['max_value'], step=kwargs['step_value'], description='z [m]')
    display(zvalue)
    link = wg.jslink((zvalue, 'value'), (zslider, 'value'))
    
    wg.interact(plot_caustic_slice, z=zslider, 
                filename=wg.fixed(kwargs['filename']), dataset_prefix=wg.fixed(kwargs['dataset_prefix']), 
                plot=wg.fixed(kwargs['plot']), unitFactor=wg.fixed(kwargs['unitFactor']), 
                xlabel=wg.fixed(kwargs['xlabel']), ylabel=wg.fixed(kwargs['ylabel']), 
                units=wg.fixed(kwargs['units']), xlim=wg.fixed(kwargs['xlim']), ylim=wg.fixed(kwargs['ylim']), 
                scale=wg.fixed(kwargs['scale']), showPlot=wg.fixed(kwargs['showPlot']), showFWHM=wg.fixed(kwargs['showFWHM']))






