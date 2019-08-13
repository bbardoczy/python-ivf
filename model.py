#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main file

"""

import numpy as np
from between_states import ev_emu


class model:
    def __init__(self,pars,iterator_name="EGM"):
        from grids_and_functions import setupClass
        self.setup = setupClass(pars)
        self.iterator_name = iterator_name
        self.initialize, self.iterate = self.get_iterator(iterator_name)
        
        
    # this sets up iterators and initializers
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
        
        
    
    
    # this actually solves the model for V
    def compute_V(self):
        self.V =  [{ 'No children':None, 'One child, out':None, 'One child, in':None }]*self.setup.pars['T']
        self.descriptions = [*self.V[0]]
        
        T = self.setup.pars['T']
        
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
            
            
                
    # this fits output of iterate
    def vpack(self,Vcs,time,desc):
        from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete
        agrid = self.setup.agrid
        zgsgrid = self.setup.zgs_Grids[desc][time]        
        Vout = valuefunction([grid_dim_linear(agrid),grid_dim_discrete(zgsgrid)],{'V':Vcs[0],'c':Vcs[1],'s':Vcs[2]},time,desc)
        return Vout
    

