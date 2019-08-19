#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 27 13:39:24 2019

@author: Egor Kozlov

This thing runs the main calculations for IVF project.

"""

#import numpy as np

from model import Model, Agents
from timeit import default_timer
#from numba import jit
    


def model_solve(pars):#,return_am=False):
    
    
        M = Model(pars,"EGM")
        M.compute_V()
        #V = M.V
        #M.refactoring_check()
        
        a = Agents(M)
        a.simulate()
        
        return_am=False
        
        if return_am:
            return a, M
        else:
            return a.state.mean(axis=0)
        
# this runs the file    
if __name__ == "__main__":
    
    start = default_timer()

    pars_in = dict(
                    T = 20,
                    sigma = 2,
                    R = 1.03,
                    beta = 0.95,
                    g_kid = 0.05,
                    z_kid = 0.2,
                    u_kid_c = 0.05,
                    u_kid_add = 0.05,
                    phi_in = 1,
                    phi_out = 0.4,
                    pback = 0.25,
                    eps = 0.00,
                    amax = 40, na = 200,
                    a_z0 = 0.0, a_z1 = 0.00, a_z2 = 0.00,
                    a_g0 = 0.0, a_g1 = -0.1,
                    sigma_z_init=0.15,sigma_z=0.1,nz=7,
                    sigma_g=0.1,rho_g=0.8,ng=7,smax=4.12,ns=16
                  )
    
    #@jit
    
    from multiprocessing import Pool
    from scoop import futures

    pool = Pool(4)
    
    
    
    nit = 4
    plist = [pars_in.copy() for i in range(nit)]
    
    # NB: this is a shallow copy. Make sure all elements are primitive and not links
    
    
    
    
    for i in range(nit):
        plist[i]['u_kid_add'] = plist[0]['u_kid_add'] + 0.01*i
        
    
    
    ser_start = default_timer()
    g = [model_solve(p) for p in plist]
    ser_finish = default_timer()
    print('Serial time is {} seconds'.format( round(ser_finish-ser_start, 2)) )
    
    
    
    par_start = default_timer()
    g = pool.map(model_solve, plist)
    par_finish = default_timer()
    print('Parallel time is {} seconds'.format( round(par_finish-par_start,2) ))
    
    
    scoop_start = default_timer()
    g = list(futures.map(model_solve, plist))
    scoop_finish = default_timer()
    print('Scoop time is {} seconds'.format( round(scoop_finish-scoop_start,2) ))
    
    
    
    finish = default_timer()
    print( 'Total time is {} seconds'.format( round(finish - start,2) ) )
    
    
    pool.close()
    del pool
    