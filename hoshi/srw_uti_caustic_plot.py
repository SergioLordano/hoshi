#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:44:18 2018

@author: sergio.lordano
"""

from optlnls.math import get_fwhm
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def read_caustic(filename_h5='test.h5'):
    
    f = h5py.File(filename_h5, 'r')
    
    odict = dict()
    
    odict['xmin'] = f.attrs['xStart']
    odict['xmax'] = f.attrs['xFin']
    odict['nx'] = f.attrs['nx']
    odict['ymin'] = f.attrs['yStart']
    odict['ymax'] = f.attrs['yFin']
    odict['ny'] = f.attrs['ny']
    odict['emin'] = f.attrs['eStart']
    odict['emax'] = f.attrs['eFin']
    odict['ne'] = f.attrs['ne']
    odict['zOffset'] = f.attrs['zOffset']
    odict['zmin'] = f.attrs['zStart'] + odict['zOffset'] 
    odict['zmax'] = f.attrs['zFin'] + odict['zOffset'] 
    odict['nz'] = f.attrs['nz']
    odict['zStep'] = f.attrs['zStep']

           
    caustic3D = np.zeros((odict['nz'], odict['ny'], odict['nx']))
    
    integral = np.zeros(odict['nz'])
    max_intensity = np.zeros(odict['nz'])
    
    for i in range(odict['nz']):
        dataset = 'step_{0}'.format(i)
        caustic3D[i] = np.array(f[dataset])
        integral[i] = f[dataset].attrs['integral']
        max_intensity[i] = f[dataset].attrs['max intensity']
        
    f.close()
        
    odict['caustic'] = caustic3D 
    odict['integral'] = integral
    odict['max intensity'] = max_intensity        
    
    return odict
    
def get_x_cut(caustic_dict, projected=1, ypos=0):
       
    cdict = caustic_dict
    
    if(projected):        
        cut = np.sum(cdict['caustic'], axis=1).transpose()
        y0 = np.nan
    else:
        y = np.linspace(cdict['ymin'], cdict['ymax'], cdict['ny'])
        idx = np.abs(y - ypos).argmin()
        y0 = y[idx]
        if((ypos < np.min(y)) or (ypos > np.max(y))):
            print('   *** WARNING: ypos out of range [', cdict['ymin'], ',', cdict['ymax'], ']')
        cut = cdict['caustic'][:,idx,:].transpose()
                
    return [cut, y0]

def get_y_cut(caustic_dict, projected=1, xpos=0):
       
    cdict = caustic_dict
    
    if(projected):        
        cut = np.sum(cdict['caustic'], axis=2).transpose()
        x0 = np.nan
    else:
        x = np.linspace(cdict['xmin'], cdict['xmax'], cdict['nx'])
        idx = np.abs(x - xpos).argmin()
        x0 = x[idx]
        if((xpos < np.min(x)) or (xpos > np.max(x))):
            print('   *** WARNING: xpos out of range [', cdict['xmin'], ',', cdict['xmax'], ']')
        cut = cdict['caustic'][:,:,idx].transpose()
                
    return [cut, x0]

def get_xy_cut(caustic_dict, zpos=0):
    
    cdict = caustic_dict
    
    z = np.linspace(cdict['zmin'] + cdict['zOffset'], 
                    cdict['zmax'] + cdict['zOffset'], 
                    cdict['nz'])
    
    idx = np.abs(z - zpos).argmin()
    if((zpos < np.min(z)) or (zpos > np.max(z))):
        print('   *** WARNING: zpos out of range [', z.min(), ',', z.max(), ']')
    cut = cdict['caustic'][idx,:,:].transpose()
    
    return [cut, z[idx]]
    
    
   
    
def plot_caustic2D_cut(filename='test.h5', showAxis='x', fixedPosition=0.0, 
                       zUnitFactor=1.0, uUnitFactor=1e6, zLabel='Z [m]', uLabel='X [$\mu m$]', 
                       aspect='auto', ulim=None, zlim=None, clim=None, minThreshold=0, 
                       scale='linear', figsize=(8,6), fontsize=12,
                       projected=1, showPlot=1, savefig=1):
    
    cdict = read_caustic(filename)
    
    xmin = cdict['xmin']
    xmax = cdict['xmax']
    ymin = cdict['ymin']
    ymax = cdict['ymax']
    zOffset = cdict['zOffset']
    zmin = cdict['zmin'] + zOffset
    zmax = cdict['zmax'] + zOffset
    
    #### define axis (u)    
    if(showAxis == 'x'):
        uz, y0 = get_x_cut(cdict, projected, fixedPosition)
        posfix = '_xz' 
        umin = xmin
        umax = xmax

    if(showAxis == 'y'):
        uz, x0 = get_y_cut(cdict, projected, fixedPosition)    
        posfix = '_yz' 
        umin = ymin
        umax = ymax
        
    #### define limits and extent
    if(minThreshold > 0):
        uz_min, uz_max = np.max(uz)*minThreshold, np.max(uz)
    
    else:    
        if(clim is not None):
            uz_min, uz_max = clim[0], clim[1]
        else:
            uz_min, uz_max = np.min(uz), np.max(uz)
        
    extent = [zmin*zUnitFactor, zmax*zUnitFactor, 
              umin*uUnitFactor, umax*uUnitFactor]
        
    #### create plot
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(0.1,0.18,0.99,0.95)
    
    if(scale=='linear'):
        plt.imshow(uz, aspect=aspect, origin='lower', 
                   vmin=uz_min, vmax=uz_max, extent=extent)
    
    elif(scale=='log'):
        if(uz_min <= 0.0):
            uz_min_except_0 = np.min(uz[uz>0])
            uz[uz<=0.0] = uz_min_except_0/2.0
            uz_min = uz_min_except_0/2.0                    
        plt.imshow(uz, aspect=aspect, origin='lower', 
                   norm=LogNorm(vmin=uz_min, vmax=uz_max),
                   extent=extent)
    
    plt.ylabel(uLabel, fontsize=fontsize)
    plt.xlabel(zLabel, fontsize=fontsize)
        
    if zlim is not None:
        plt.xlim(zlim[0], zlim[1])
    if ulim is not None:
        plt.ylim(ulim[0], ulim[1])
    
    ax = plt.gca()
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=fontsize)
    plt.minorticks_on()
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
            
    if(savefig):
        plt.savefig(filename[:-3]+posfix+'.png', dpi=400)    
    
    if(showPlot):
        plt.show()
    
    return cdict


def test():
    
    global caustic_dict, cut
    
    fname = '/media/lordano/DATA/LNLS/Oasys/EMA/XFW/XFW_FZP/ThinLens_annular_caustic.h5'

    cdict = read_caustic(fname)
    

    
    # plot_caustic2D_cut(filename=fname, showAxis='x', projected=1, savefig=0,
    #                    scale='log', minThreshold=0, clim=[1e16,1e20])
    
    # cut = get_y_cut(cdict, projected=0)    
    xy, zpos = get_xy_cut(cdict, zpos=0.52)
    
    plt.figure()
    plt.imshow(xy, aspect='auto')
    
    z = np.linspace(cdict['zmin'] + cdict['zOffset'], 
                    cdict['zmax'] + cdict['zOffset'], 
                    cdict['nz'])
    max_int = cdict['max intensity']
    
    plt.figure()
    plt.plot(z, max_int)
    plt.yscale('log')


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


if __name__ == '__main__':
    
    # test()
    pass

