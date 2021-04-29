#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:15:24 2018

@author: sergio.lordano
"""


def SRW_beam_caustic(wfr=None, zStart=1.0, zFin=2.0, zStep=0.1, zOffset=0.0, extract_parameter=1, useMPI=True, save_hdf5=False, h5_filename='test.h5', buffer=False, matrix=True, ppFin=None):
    """
    :extract_parameter: (1) Total Intensity, (2) Hor. Pol. Intensity, (3) Vert. Pol. Intensity, (4) Hor. Pol. Phase, (5) Hor. Pol. Phase, (6) Hor. Pol. Electric Field, (7) Hor. Pol. Electric Field
    """
    
    import numpy as np
    from srwlib import SRWLOptA, SRWLOptD, SRWLOptC, SRWLWfr
    import srwlpy as srwl
    from array import array
    import time
    import copy
    from scipy.integrate import simps
    
    if(save_hdf5):
        import h5py
    
    #############################################
    #### INITIALIZE THREADS                   ###
    #############################################

    MPI=None
    comMPI=None
    nProc=1
    rank=0
    
    if(useMPI):
        
        from mpi4py import MPI
        comMPI = MPI.COMM_WORLD
        nProc = comMPI.Get_size() # total number of threads
        rank = comMPI.Get_rank() # particular thread executing

    if(wfr==None):
        
        wfr = SRWLWfr()

    t0 = time.time()    
    
    #############################################
    #### DEFINE THE POSITIONS TO CALCULATE IN ###
    #############################################

    nz = int((zFin-zStart)/zStep+1) # total number of positions
    positions = np.linspace(zStart, zFin, nz) # array of positions

    #############################################
    #### DEFINE THE FUNCTION TO BE CALLED     ###
    #############################################
    
    def propagate_distance(wfri, distance, extract_parameter, ppFin):
        
        PP_GEN = [ 0, 0, 1.5, 0, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
        
        if ppFin is not None:
            optBLj = SRWLOptC([SRWLOptD(distance)], [PP_GEN, ppFin])
        else:
            optBLj = SRWLOptC([SRWLOptD(distance)], [PP_GEN])
        
        wfrj = copy.deepcopy(wfri)
        
        if(distance != 0.0): # if distance is zero, the wavefront is not propagated
            srwl.PropagElecField(wfrj, optBLj)
        
        if(distance == 0.0 and ppFin is not None): # if distance is zero
            srwl.PropagElecField(wfrj, SRWLOptC([SRWLOptA('c', 'a', 10.0)], [ppFin]))
        
        if(extract_parameter==1): # intensity: 2D, polarization: total
            arI = array('f', [0]*wfrj.mesh.nx*wfrj.mesh.ny)
            srwl.CalcIntFromElecField(arI  , wfrj, 6, 0, 3, wfrj.mesh.eStart, 0, 0)
            return np.array(arI, dtype=np.float64), wfrj
        
        elif(extract_parameter==2): # intensity: 2D, polarization: x
            arIx = array('f', [0]*wfrj.mesh.nx*wfrj.mesh.ny) #"flat" 2D array to take intensity data
            srwl.CalcIntFromElecField(arIx, wfrj, 0, 0, 3, wfrj.mesh.eStart, 0, 0) # Intensity from electric field (Wfr_name, Disp_Name, Polariz., FuncOf, Extr, ObsEi , ConstHorPos, ConstVerPos, NewDisp)
            return np.array(arIx, dtype=np.float64), wfrj
            
        elif(extract_parameter==3): # intensity: 2D, polarization: y
            arIy = array('f', [0]*wfrj.mesh.nx*wfrj.mesh.ny) #"flat" 2D array to take intensity data
            srwl.CalcIntFromElecField(arIy, wfrj, 1, 0, 3, wfrj.mesh.eStart, 0, 0) # Intensity from electric field (Wfr_name, Disp_Name, Polariz., FuncOf, Extr, ObsEi , ConstHorPos, ConstVerPos, NewDisp)
            return np.array(arIy, dtype=np.float64), wfrj
            
        elif(extract_parameter==4): # phase: 2D, polarization: x
            arPx = array('d', [0]*wfrj.mesh.nx*wfrj.mesh.ny) #"flat" array to take 2D phase data (note it should be 'd')
            srwl.CalcIntFromElecField(arPx, wfrj, 0, 4, 3, wfrj.mesh.eStart, 0, 0) #extracts radiation phase
            return np.array(arPx, dtype=np.float64), wfrj
                
        elif(extract_parameter==5): # phase: 2D, polarization: y
            arPy = array('d', [0]*wfrj.mesh.nx*wfrj.mesh.ny) #"flat" array to take 2D phase data (note it should be 'd')
            srwl.CalcIntFromElecField(arPy, wfrj, 1, 4, 3, wfrj.mesh.eStart, 0, 0) #extracts radiation phase
            return np.array(arPy, dtype=np.float64), wfrj
    
    def integrate_array2D(array2D, xStart, xFin, nx, yStart, yFin, ny):
        
        x_axis = np.linspace(xStart, xFin, nx)*1e3 # work in [mm]
        y_axis = np.linspace(yStart, yFin, ny)*1e3 # work in [mm]
        
        int_y = np.zeros((ny)) # array to store the integrated array over x_axis
        for i in range(ny):
            int_y[i] = simps(array2D[i, :], x=x_axis)
        
        integral = simps(int_y, x=y_axis)
        return integral
        
    
    def initialize_hdf5(wfr0):
        
        with h5py.File(h5_filename, 'w') as f:
                
                f.attrs['begin time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.attrs['xStart'] = wfr0.mesh.xStart
                f.attrs['xFin'] = wfr0.mesh.xFin
                f.attrs['nx'] = wfr0.mesh.nx
                f.attrs['yStart'] = wfr0.mesh.yStart
                f.attrs['yFin'] = wfr0.mesh.yFin
                f.attrs['ny'] = wfr0.mesh.ny
                f.attrs['eStart'] = wfr0.mesh.eStart
                f.attrs['eFin'] = wfr0.mesh.eFin
                f.attrs['ne'] = wfr0.mesh.ne
                f.attrs['zStart'] = zStart
                f.attrs['zFin'] = zFin
                f.attrs['nz'] = nz
                f.attrs['zOffset'] = zOffset
                f.attrs['zStep'] = zStep
                f.attrs['extract_parameter'] = extract_parameter
                if(matrix):
                    f.attrs['format'] = 'array2D'
                else:
                    f.attrs['format'] = 'array1D'
    
    def append_dataset_hdf5(data, tag, t0):
        
        with h5py.File(h5_filename, 'a') as f:
            
            xStart = f.attrs['xStart']
            xFin = f.attrs['xFin']
            nx = f.attrs['nx']
            yStart = f.attrs['yStart']
            yFin = f.attrs['yFin']
            ny = f.attrs['ny']
            integral = integrate_array2D(data, xStart, xFin, nx, yStart, yFin, ny)
            
            dset = f.create_dataset('step_{0}'.format(tag), data=data, compression="gzip")
            dset.attrs['z'] = positions[tag] + zOffset
            dset.attrs['ellapsed time (s)'] = round(time.time() - t0, 3)
            dset.attrs['max intensity'] = np.max(data)
            dset.attrs['integral'] = integral
            
            if(tag == nz-1): 
                f.attrs['end time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                
    def write_max_peak_int(h5_filename):
    
        with h5py.File(h5_filename, 'r+') as f:
            
            max_peak_int = [0.0, 0.0]
            max_integral = [0.0, 0.0]
            
            for dset_name in f:
    
                max_dset = f[dset_name].attrs['max intensity']
                if(max_dset > max_peak_int[0]):
                    max_peak_int[0] = max_dset
                    max_peak_int[1] = f[dset_name].attrs['z'] 
                
                int_dset = f[dset_name].attrs['integral']
                if(int_dset > max_integral[0]):
                    max_integral[0] = int_dset
                    max_integral[1] = f[dset_name].attrs['z']
    
            f.attrs['max peak int'] = max_peak_int[0]
            f.attrs['max peak z'] = max_peak_int[1]
            f.attrs['max integral'] = max_integral[0]
            f.attrs['max integral z'] = max_integral[1]
    
    ###################################################
    #### DEFINE VARIABLES FOR PARALLEL CALCULATIONS ###
    ###################################################
    
    if(nProc > 1):
        if(nz % (nProc-1) != 0): # Increases 1 round if nz is not divisible by n_threads       
            n_rounds = int(nz/(nProc-1))+1 # number of times that each rank will execute
        else:
            n_rounds = int(nz/(nProc-1)) # number of times that each rank will execute
    else:
        n_rounds = nz
        
    #############################################
    #### START RUNNING IN PARALLEL MODE       ###
    #############################################
    
    if(nProc > 1 and rank == 0): # master process saving data received from slaves to hdf5
               
        for i_iteration in range(1, nz+1):

            if(i_iteration==1):
                arIP, wfri = propagate_distance(wfr, 0.0, extract_parameter, ppFin)
                if(save_hdf5): # initialize file
                    initialize_hdf5(wfri)  
            
            data_recv = np.zeros(wfr.mesh.nx*wfr.mesh.ny, dtype=np.float64) # allocate array
            
            status = MPI.Status() 
            comMPI.Recv([data_recv, MPI.FLOAT], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            
            if(matrix):
                data_recv = data_recv.reshape((wfri.mesh.ny,wfri.mesh.nx)) # reshape array
            
            tag = status.Get_tag()             
            
            if(save_hdf5):
                append_dataset_hdf5(data_recv, tag, t0)
        
        write_max_peak_int(h5_filename) # add max intensity attributes to group        
        return [0]
        
    elif(nProc > 1 and rank > 0): # slave processes sending data to master
        
        for i_round in range(n_rounds): # iterate over rounds

            i_value = rank + i_round*(nProc-1) - 1 # get actual value

            if(i_value < nz): # check if n_total was not exceeded
                
                arIP = np.empty(wfr.mesh.nx*wfr.mesh.ny, dtype=np.float64) # allocate array
                arIP, wfrm = propagate_distance(wfr, positions[i_value], extract_parameter, ppFin) # Propagate wavefront to each distance 
                comMPI.Send([arIP, MPI.FLOAT], dest=0, tag=i_value) # send to master with tag number
                
        return [0]
    
    #############################################
    #### START RUNNING IN SERIAL MODE         ###
    #############################################
    
    elif(nProc == 1): # single process doing everything (serial calculation)
        

        arIP, wfri = propagate_distance(wfr, 0.0, extract_parameter, ppFin)
        if(save_hdf5): # initialize file
            initialize_hdf5(wfri)  
        
        for i_round in range(n_rounds): # iterate over rounds
                
            data, wfrm = propagate_distance(wfr, positions[i_round], extract_parameter, ppFin) # Propagate wavefront to each distance
            if(matrix):
                data = data.reshape((wfri.mesh.ny, wfri.mesh.nx))
                
            append_dataset_hdf5(data, i_round, t0)
            
        write_max_peak_int(h5_filename) # add max intensity attributes to group
        
        
    

#SRW_beam_caustic()
