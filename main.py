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
from numpy import log
import numpy as np
#from numba import jit
    
#@jit
def model_solve(pars,return_am=False):
    
        M = Model(pars,"EGM")
        M.compute_V()
        #V = M.V
        #M.refactoring_check()
        
        a = Agents(M)
        a.simulate()
        
        
        if return_am:
            print(a.state.mean(axis=0))
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
                    g_kid = [0.05,0.1],
                    z_kid = [0.2,0.4],
                    u_kid_c = [0.15, 0.2],
                    u_kid_add = [0.15, 0.25],
                    phi_in = 1,
                    phi_out = 0.4,
                    pmar = 0.25, 
                    pback = 0.25, delta_out = 0.15,
                    eps  = 0.01,
                    pbirth = 0.5,
                    amax = 100, na = 200,
                    a_z0 = 0.0, a_z1 = 0.00, a_z2 = 0.00,
                    a_g0 = 0.0, a_g1 = 0,
                    sigma_z_init=0.15,sigma_z=0.1,nz=7,
                    sigma_g=0.1,rho_g=0.8,ng=7,smax=log(4.12),ns=16
                  )
    
    
    
    
    
    #a, M = model_solve(pars_in,True)
    
    
    target = 'delta_out'
    
    
    #step = 5e-5
    
    #pars_in_change = pars_in.copy()
    #pars_in_change[target] = pars_in[target]+step
    a0, M0 = model_solve(pars_in,True)
    #a1, M1 = model_solve(pars_in_change,True)
    
    #print('Changed for {} of agents'.format(np.mean(np.any(a0.state!=a1.state,axis=1))))
    
    
    
    finish = default_timer()
    print( 'Total time is {} seconds'.format( round(finish - start,2) ) )