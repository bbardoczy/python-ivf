#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 27 13:39:24 2019

@author: Egor Kozlov

This thing runs the main calculations for IVF project.

"""

#import numpy as np

from model import Model, Agents
        
    
# this runs the file    
if __name__ == "__main__":
    
    pars_in = dict(
                    T = 40,
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
                    eps = 0.05,
                    amax = 50, na = 200,
                    a_z0 = 0.0, a_z1 = 0.00, a_z2 = 0.00,
                    a_g0 = 0.0, a_g1 = -0.1,
                    sigma_z_init=0.15,sigma_z=0.1,nz=7,
                    sigma_g=0.1,rho_g=0.8,ng=7,smax=4.12,ns=16
                  )
    
    T = pars_in['T']
    
    # solve the model
    M = Model(pars_in,"EGM")
    M.compute_V()
    V = M.V
    #M.refactoring_check()
    
    a = Agents(M)
    a.simulate()
    
    