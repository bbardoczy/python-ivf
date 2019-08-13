#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 27 13:39:24 2019

@author: Egor Kozlov

This thing runs the main calculations for IVF project.

"""

import matplotlib.pyplot as plt
import numpy as np
from between_states import ev_emu



class model:
    def __init__(self,pars,iterator_name="EGM"):
        from grids_and_functions import setupClass
        self.setup = setupClass(pars)
        self.iterator_name = iterator_name
        self.initialize, self.iterate = self.get_iterator(iterator_name)
        
        
    # this sets up iterators    
    def get_iterator(self,name):
        if name == "EGM":
            
            from vf_iterators import Vnext_egm
            s = self.setup
            def iterate(desc,t,EV,EMU):
                gri, ma = s.zgs_Grids[desc][t], s.zgs_Mats[desc][t]            
                return Vnext_egm(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])
        
            def initialize(desc,T):
                return iterate(desc,T,None,None)
            
        elif name == "VFI":
            
            from vf_iterators import Vnext_vfi
            s = self.setup
            def iterate(desc,t,EV,EMU):
                gri, ma = s.zgs_Grids[desc][t], s.zgs_Mats[desc][t]            
                return Vnext_vfi(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])
            def initialize(desc,T):
                return iterate(desc,T,None,None)
            
        else:
            raise Exception('Cannot find iterator!')
        
        return initialize, iterate
        
        
        
    def compute_V(self):
        self.V =  [{ 'No children':None, 'One child, out':None, 'One child, in':None }]*self.setup.pars['T']
        self.descriptions = [*self.V[0]]
        
        # this part is backward iteration
        for t in reversed(range(T)):
            
            for desc in self.descriptions:
                if t == T-1:
                    Vcs = self.initialize(desc,t)  
                else:
                    s = self.setup
                    ma = s.zgs_Mats[desc][t]            
                    integrate = lambda V : np.dot(V,ma.T)            
                    Vcomb, MU_comb = ev_emu(self.V[t+1],s.transitions[desc],mu=s.mu[desc])                
                    EV, EMU  = integrate(  Vcomb  ), integrate(  MU_comb  )  
                    Vcs = self.iterate(desc,t,EV,EMU)                
                    assert np.all(MU_comb > 0)  
        
                self.V[t][desc] = self.vpack(Vcs,T-1,desc)
            
            
                
    
    def vpack(self,Vcs,time,desc):
        from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete
        agrid = self.setup.agrid
        zgsgrid = self.setup.zgs_Grids[desc][time]        
        Vout = valuefunction([grid_dim_linear(agrid),grid_dim_discrete(zgsgrid)],{'V':Vcs[0],'c':Vcs[1],'s':Vcs[2]},time,desc)
        return Vout
    


        




    
# this runs the file    
if __name__ == "__main__":
    
    
    # imports
    
    
    pars_in = dict(
                    T = 20,
                    sigma = 1,
                    R = 1.03,
                    beta = 0.97,
                    g_kid = 0.05,
                    z_kid = 0.2,
                    u_kid_c = 0.0,
                    u_kid_add = 0.05,
                    phi_in = 1,
                    phi_out = 0.4,
                    pback = 0.25,
                    eps = 0.00,
                    amax = 40, na = 200,
                    a_z0 = 0.0, a_z1 = 0.05, a_z2 = 0.05,
                    a_g0 = 0.0, a_g1 = -0.1,
                    sigma_z_init=0.15,sigma_z=0.1,nz=7,
                    sigma_g=0.1,rho_g=0.8,ng=7,smax=4.12,ns=16
                )
    
    T = pars_in['T']
    
    
    # set iterator
    M = model(pars_in,"EGM")
    M.compute_V()
    V = M.V
    
    # initialize V
    
    it = 0
    
    try:
        assert np.abs(V[it]["No children"][5,1000] - 60.043482353008635)<1e-10
        assert np.abs(V[it]["One child, in"][5,1000]-V[it]["No children"][5,1000] + 4.6052711813603295) < 1e-10
        assert np.abs(V[it]["One child, out"][5,1000]-V[it]["No children"][5,1000] + 7.95700682295378) < 1e-10
        print('Tests are ok!')
    except:   
        # draw things and print diagnostic informatio
        print(V[it]["No children"][5,1000])
        print(V[it]["One child, in"][5,1000]-V[it]["No children"][5,1000])
        print(V[it]["One child, out"][5,1000]-V[it]["No children"][5,1000])
        plt.cla()
        plt.subplot(211)
        V[it][  "No children"  ].plot_value( ['s',['s','c',np.add],np.divide] )
        V[it][  "One child, in"].plot_value( ['s',['s','c',np.add],np.divide] )
        plt.subplot(212)
        V[it]["No children"].plot_diff(V[it]["One child, in"],['s',['s','c',np.add],lambda x, y: np.divide(x,np.maximum(y,1)) ])
        plt.legend()
        raise Exception('Tests are not ok!')
    
    
    
        
    
    