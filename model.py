#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main file

"""


import numpy as np
from between_states import ev_emu
from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete
from grids_and_functions import setupClass


class model:
    def __init__(self,pars,iterator_name="EGM"):
        
        
        self.setup = setupClass(pars)
        self.iterator_name = iterator_name
        self.initialize, self.iterate = self.get_iterator(iterator_name)
        
        
    # this sets up iterators and initializers
    def get_iterator(self,name):
        
        # this packs output of an iterator to valuefunction class
        def vpack(Vcs,time,desc):
            
            agrid = self.setup.agrid
            zgsgrid = self.setup.zgs_Grids[desc][time]        
            
            grids = [grid_dim_linear(agrid),grid_dim_discrete(zgsgrid)]
            
            Vout = valuefunction(grids,{'V':Vcs[0],'c':Vcs[1],'s':Vcs[2]},time,desc)
            
            return Vout
        
        
        # this actually selects the iterator
        if name == "EGM":
            
            from vf_iterators import Vnext_egm
            s = self.setup
            
            def iterate(desc,t,EV,EMU):
                gri, ma = s.zgs_Grids[desc][t], s.zgs_Mats[desc][t]            
                Vcs = Vnext_egm(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])
                return vpack(Vcs,t,desc)
        
            def initialize(desc,t):
                return iterate(desc,t,None,None)
            
        elif name == "VFI":
            
            from vf_iterators import Vnext_vfi
            s = self.setup
            def iterate(desc,t,EV,EMU):
                gri, ma = s.zgs_Grids[desc][t], s.zgs_Mats[desc][t]    
                Vcs = Vnext_vfi(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])
                return vpack(Vcs,t,desc)
            def initialize(desc,t):
                return iterate(desc,t,None,None)
            
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
                    #Vcs = self.initialize(desc,t)  
                    self.V[t][desc] = self.initialize(desc,t)  
                else:
                    s = self.setup
                    ma = s.zgs_Mats[desc][t]            
                    integrate = lambda V : np.dot(V,ma.T)            
                    Vcomb, MU_comb = ev_emu(self.V[t+1],s.transitions[desc],mu=s.mu[desc])                
                    EV, EMU  = integrate(  Vcomb  ), integrate(  MU_comb  )  
                    #Vcs = self.iterate(desc,t,EV,EMU)
                    self.V[t][desc] = self.iterate(desc,t,EV,EMU)
                    assert np.all(MU_comb > 0)  
        
                #self.V[t][desc] = self.vpack(Vcs,t,desc)
            
            
                
    # this fits output of iterate
    
    
    def refactoring_check(self):
        
        V = self.V
        it = 0
    
        try:
            assert np.abs(V[it]["No children"][5,1000] - 60.043482353008635)<1e-10
            assert np.abs(V[it]["One child, in"][5,1000]-V[it]["No children"][5,1000] + 4.6052711813603295) < 1e-10
            assert np.abs(V[it]["One child, out"][5,1000]-V[it]["No children"][5,1000] + 7.95700682295378) < 1e-10
            print('Tests are ok!')
        except:   
            import matplotlib.pyplot as plt

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
    
    

