#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:25:05 2020

@author: lordano
"""


import numpy as np
from matplotlib import pyplot as plt

def read_srw_int(filename):
    with open(filename, 'r') as infile:
        data = infile.readlines()
    infile.close()
    
    ei = float(data[1].split('#')[1])
    ef = float(data[2].split('#')[1])
    en = int(data[3].split('#')[1])
    xi = float(data[4].split('#')[1])
    xf = float(data[5].split('#')[1])
    xn = int(data[6].split('#')[1])
    yi = float(data[7].split('#')[1])
    yf = float(data[8].split('#')[1])
    yn = int(data[9].split('#')[1])
    
    nheaders = 11
    if not(data[10][0]=='#'): nheaders = 10
    
    if(0):       
#       #loop method      
        intensity = np.zeros((en, yn, xn))       
        count = 0     
        for i in range(yn):
            for j in range(xn):
                for k in range(en):
                    intensity[k, i, j] = data[count + nheaders]
                    count += 1
    if(1):            
#       #Reshape method
        intensity = np.array(data[nheaders:], dtype='float').reshape((en, yn, xn))
    
    return intensity

def read_srw_efield(filename, shape):
    with open(filename, 'r') as infile:
        efield = infile.readlines()
    infile.close()        

    elec_field = np.zeros((shape[0], shape[1]), dtype='complex')
    elec_field.real = np.array(efield[::2]).reshape((shape[0], shape[1]))
    elec_field.imag = np.array(efield[1::2]).reshape((shape[0], shape[1]))
    
    return elec_field

def write_srw_transmission(filename, amplitude_3D, opd_3D, energy_grid, hor_grid, vert_grid):
    import numpy as np
        
    ne = energy_grid[2]
    nx = hor_grid[2]
    ny = vert_grid[2]

    nTot = 2*ne*nx*ny        
    arTr = np.array([0]*nTot, dtype='float')
    count = 0
    for i in range(ny):
        for j in range(nx):
            for k in range(ne):
                arTr[count] = amplitude_3D[k,i,j]
                if(opd_3D is None):
                    arTr[count+1] = 0
                else:
                    arTr[count+1] = opd_3D[k,i,j]
                count += 2
                
    with open(filename,'w') as arrayfile:
    
        arrayfile.write('#' + 'Transmission - Amplitude and Optical Path Difference. ' + ' (C-aligned, inner loop is vs ' + 'energy' + ', outer loop vs ' + 'Vertical Position' + ')\n')
        arrayfile.write('#' + repr(energy_grid[0]) + ' #Initial ' + 'energy' + '\n')
        arrayfile.write('#' + repr(energy_grid[1]) + ' #Final ' + 'energy' + '\n')
        arrayfile.write('#' + repr(energy_grid[2]) + ' #Number of points vs ' + 'energy' + '\n')
        arrayfile.write('#' + repr(hor_grid[0]) + ' #Initial ' + 'Horizontal Position' + '\n')
        arrayfile.write('#' + repr(hor_grid[1]) + ' #Final ' + 'Horizontal Position' + '\n')
        arrayfile.write('#' + repr(hor_grid[2]) + ' #Number of points vs ' + 'Horizontal Position' + '\n')
        arrayfile.write('#' + repr(vert_grid[0]) + ' #Initial ' + 'Vertical Position' + '\n')
        arrayfile.write('#' + repr(vert_grid[1]) + ' #Final ' + 'Vertical Position' + '\n')
        arrayfile.write('#' + repr(vert_grid[2]) + ' #Number of points vs ' + 'Vertical Position' + '\n')
        
        for line in range(nTot):
            arrayfile.write('{0:.8e}'.format(arTr[line]))
            if(line != nTot-1):
                arrayfile.write('\n')
    arrayfile.close()
    
def read_srw_transmission(filename):
    with open(filename, 'r') as infile:
        transmission = infile.readlines()
    infile.close() 
    
    ei = float(transmission[1].split('#')[1])
    ef = float(transmission[2].split('#')[1])
    en = int(transmission[3].split('#')[1])
    xi = float(transmission[4].split('#')[1])
    xf = float(transmission[5].split('#')[1])
    xn = int(transmission[6].split('#')[1])
    yi = float(transmission[7].split('#')[1])
    yf = float(transmission[8].split('#')[1])
    yn = int(transmission[9].split('#')[1])
    
    nheaders = 11
    if not(transmission[10][0]=='#'): nheaders = 10
    
    transm = np.zeros((2, en, yn, xn), dtype='float')
    count = 0     
    for i in range(yn):
        for j in range(xn):
            for k in range(en):
                transm[0, k, i, j] = transmission[nheaders + count]
                transm[1, k, i, j] = transmission[nheaders + count+1]
                count += 2
    
    return transm, [ei, ef, en], [xi, xf, xn], [yi, yf, yn]



### == Defines an array of apertures ====================================== ###
def ApertureArray(Lx, nx, Dx, x0, transmission_factor=1.0):
    import numpy as np
    x = np.linspace(-Lx/2, Lx/2, nx, dtype='float')
    transmission = np.zeros((nx))
    for x0i in x0:
        idx_min = np.abs(x-(x0i-Dx/2.0)).argmin()
        idx_max = np.abs(x-(x0i+Dx/2.0)).argmin()
        transmission[idx_min:idx_max+1] = 1.0*transmission_factor
    return x, transmission

def ApertureArraySRW(Dx, Dy, x0):
    import numpy as np
    from srwlib import SRWLOptA
    
    ap_array = []
    
    if(len(x0)==1):
        AP_L = SRWLOptA(_shape='r', _ap_or_ob='a', _Dx=Dx, _Dy=Dy, _x=x0[0], _y=0)
        ap_array.append(AP_L)
    else:
        AP_L = SRWLOptA(_shape='r', _ap_or_ob='a', _Dx=(np.max(x0) - np.min(x0) + Dx), _Dy=Dy, _x=(np.max(x0) + np.min(x0))/2.0, _y=0)
        ap_array.append(AP_L)
        for i in range(len(x0)-1):
            OB_i = SRWLOptA(_shape='r', _ap_or_ob='o', _Dx=(x0[i+1]-x0[i] - Dx), _Dy=Dy, _x=(x0[i+1] + x0[i])/2.0, _y=0)  
            ap_array.append(OB_i)
            
    return ap_array


def PlaneWfr(amplitude, phase, xStart, xFin, nx, yStart, yFin, ny, eStart, eFin, ne, 
             zStart=10.0, polarization=[1**0.5,0**0.5], Rx=1e10, Ry=1e10):
    
    from array import array
    import numpy as np
    from srwlib import SRWLWfr
    
    wfr = SRWLWfr()
    wfr.allocate(ne, nx, ny)
    wfr.mesh.zStart = zStart
    wfr.mesh.eStart = eStart
    wfr.mesh.eFin = eFin
    wfr.mesh.xStart = xStart
    wfr.mesh.xFin = xFin
    wfr.mesh.yStart = yStart
    wfr.mesh.yFin = yFin
    
    eletric_fieldx = (amplitude*polarization[0])*np.exp(1j*phase) # [sqrt(ph/s/mm²/0.1%bw)]
    eletric_fieldy = (amplitude*polarization[1])*np.exp(1j*phase) # [sqrt(ph/s/mm²/0.1%bw)]
    
    nTot = 2*ne*nx*ny
    arEx = array('f', [0]*nTot)
    arEy = array('f', [0]*nTot)
    
    buffer = 0
    for i in range(ny):
        for j in range(nx):
            for k in range(ne):
                arEx[buffer] = eletric_fieldx.real
                arEx[buffer+1] = eletric_fieldx.imag
                arEy[buffer] = eletric_fieldy.real
                arEy[buffer+1] = eletric_fieldy.imag      
                buffer += 2
    
    wfr.arEx = arEx
    wfr.arEy = arEy
    wfr.Rx = Rx
    wfr.Ry = Ry
    
    wfr.partBeam.partStatMom1.x = 0
    wfr.partBeam.partStatMom1.y = 0
    wfr.partBeam.partStatMom1.z = 0
    wfr.partBeam.partStatMom1.xp = 0
    wfr.partBeam.partStatMom1.yp = 0
    
    return wfr


def srw_pp_prop0():
    return [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]

def srw_pp_prop1():
    return [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]

def srw_pp(propagator=0, xrange=1.0, xres=1.0, yrange=1.0, yres=1.0):
    return [0, 0, 1.0, propagator, 0, xrange, xres, yrange, yres, 0, 0, 0]


def thickness_to_transmission(thickness, energy, material, density=0, delta=0, beta=0):
        
    try:        
        import xraylib as xrl
        Z = xrl.SymbolToAtomicNumber(material)
        density = xrl.ElementDensity(Z)
        delta = 1 - xrl.Refractive_Index_Re(material, energy*1e-3, density)
        beta = xrl.Refractive_Index_Im(material, energy*1e-3, density)
        print("Using xraylib")
        
    except:
        print("Not using xraylib. Please provide density, delta and beta")
        
    hc = 1.2398419843320028*1e-6
    pi = 3.141592653589793
    
    wavelength = hc / energy # meters
    wavenumber = 2 * pi / wavelength  
    
    linear_absorption_coeff = 4 * pi * beta / wavelength # m-1
    attenuation_length = 1 / linear_absorption_coeff
    print(beta)
    print(linear_absorption_coeff)
    print(attenuation_length)
    
    amplitude = np.sqrt( np.exp( -1*linear_absorption_coeff * thickness ) )
    OPD = delta * thickness
    phase = wavenumber * OPD
    complex_transmission = amplitude * np.exp(1j * phase)
    
    return np.array([complex_transmission, amplitude, phase, OPD])
    
    
def image_to_thickness(filename='', max_thickness=1, rgb_channel=0, invert=0):
    
    img = plt.imread(filename)
    img = np.array(img, dtype=float)
    
    if(len(img.shape) > 2):
        img = img[:,:,rgb_channel]
        
    if(invert):
        img /= -np.max(img)
        img += 1.0        
    else:
        img /= np.max(img)
    
    img *= max_thickness
    
    return img











