#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main file

"""


import numpy as np
from between_states import ev_emu
from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete
from grids_and_functions import setupClass
from vf_iterators import Vnext_egm, Vnext_vfi
from interp_my import interpolate_nostart
from mc_tools import mc_simulate


class Model:
    def __init__(self,pars,iterator_name="EGM"):
        
        
        self.setup = setupClass(pars)
        self.iterator_name = iterator_name
        self.initialize, self.iterate = self.__get_iterator__(iterator_name)
        
        
    # this sets up iterators and initializers
    def __get_iterator__(self,name):
        
        # this packs output of an iterator to valuefunction class
        def vpack(Vcs,time,desc):
            
            agrid = self.setup.agrid
            zgsgrid = self.setup.zgs_Grids[desc][time]        
            
            grids = [grid_dim_linear(agrid),grid_dim_discrete(zgsgrid)]
            
            Vout = valuefunction(grids,{'V':Vcs[0],'c':Vcs[1],'s':Vcs[2]},time,desc)
            
            return Vout
        
        
        
        #
        # this actually selects the iterator
        #
        if name == "EGM":
            
            
            s = self.setup
            
            def iterate(desc,t,EV,EMU):
                gri, ma = s.zgs_Grids[desc][t], s.zgs_Mats[desc][t]            
                Vcs = Vnext_egm(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])
                return vpack(Vcs,t,desc)
        
            def initialize(desc,t):
                return iterate(desc,t,None,None)
            
        elif name == "VFI":
            
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
        #self.V =  [{ 'No children':None, 'One child, out':None, 'One child, in':None }]*self.setup.pars['T']
        self.V = list()
        self.descriptions = ['No children', 'One child, out', 'One child, in']#[*self.V[0]]
        
        T = self.setup.pars['T']
        
        
        integrate = lambda V, mat : np.dot(V,mat.T)   
        
        # this part is backward iteration
        for t in reversed(range(T)):
            Vthis = { 'No children':None, 'One child, out':None, 'One child, in':None }
            
            
            
            for desc in self.descriptions:
                if t == T-1:
                    #Vcs = self.initialize(desc,t)  
                    Vthis[desc] = self.initialize(desc,t)  
                else:
                    Vnext = self.V[0]
                    s = self.setup
                    ma = s.zgs_Mats[desc][t]            
                             
                    Vcomb, MU_comb = ev_emu(Vnext,s.transitions[desc],mu=s.mu[desc])                
                    EV, EMU  = integrate(  Vcomb , ma ), integrate(  MU_comb , ma )  
                    #Vcs = self.iterate(desc,t,EV,EMU)
                    Vthis[desc] = self.iterate(desc,t,EV,EMU)
                    
                    #assert not np.all(self.V[t][desc]==self.V[t+1][desc])
                    assert np.all(MU_comb > 0)  
        
        
            self.V = [Vthis] + self.V
                #self.V[t][desc] = self.vpack(Vcs,t,desc)
            
    
    
    
    
    def refactoring_check(self):
        
        V = self.V
        it = 0
    
        try:
            assert np.abs(V[it]["No children"][5,1000] - 60.04361005581497)<1e-10
            assert np.abs(V[it]["One child, in"][5,1000]-V[it]["No children"][5,1000] + 4.605243004294415) < 1e-10
            assert np.abs(V[it]["One child, out"][5,1000]-V[it]["No children"][5,1000] + 7.956746814857432 ) < 1e-10
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
    
    
class Agents:
    def __init__(self,M,N=1000,T=None):
        if T is None:
            T = M.setup.pars['T']
        self.iassets = np.zeros((N,T),np.int32)
        
        self.wassets = np.ones((N,T),np.float32)
        self.iexo = np.zeros((N,T),np.int32)
        self.iexo[:,0] = np.random.randint(0,1000,size=N)
        #self.iassets[:,0] = np.random.randint(0,100,size=N)
        
        self.state = np.zeros((N,T),np.int32)
        self.M = M
        self.V = M.V
        self.setup = M.setup
        self.state_names = list(self.V[0].keys())
        self.num_states = len(self.state_names)
        self.N = N
        self.T = T
        
        self.state_codes = {}
        for i in range(self.num_states):
            self.state_codes[self.state_names[i]] = i
        
        
        
    def anext(self,t):
        # first let's find out savings
        
        for ist in range(self.num_states):
            is_state = (self.state[:,t]==ist)
            
            nst = np.sum(is_state)
            if nst==0:
                continue
            
            ind = np.where(is_state)
            sname = self.state_names[ist]
            
            agrid = self.V[t][sname].grids[0].points
            
            s_i  = self.V[t][sname]['s'][self.iassets[ind,t],  self.iexo[ind,t]].reshape(nst)
            s_ip = self.V[t][sname]['s'][self.iassets[ind,t]+1,self.iexo[ind,t]].reshape(nst)
            w_i = self.wassets[ind,t].reshape(nst)
            anext_val =  w_i*s_i + (1-w_i)*s_ip 
            assert np.all(anext_val >= 0)
            #print((agrid.shape[0],anext_val.shape[0]))
            try:
                self.iassets[ind,t+1], self.wassets[ind,t+1], atest = interpolate_nostart(agrid,anext_val,agrid,xd_ordered=False)
            except:
                print(t,agrid.shape,anext_val.shape, w_i, s_i, s_ip)
                raise Exception('aa')
                
            assert np.all(np.abs(atest - anext_val) < 1e-6)
    
    def iexonext(self,t):
        # let's find out new state
        for ist in range(self.num_states):
            is_state = (self.state[:,t]==ist)
            nst = np.sum(is_state)
            
            if nst == 0:
                continue
            
            ind = np.where(is_state)[0]
            sname = self.state_names[ist]
            
            mat = self.setup.zgs_Mats[sname][t]
            
            iexo_now = self.iexo[ind,t].reshape(nst)
            iexo_next = mc_simulate(iexo_now,mat,shocks=None) # todo: add shocks
            
            self.iexo[ind,t+1] = iexo_next
            
            
            
    def state_next(self,t):
        
        Vnext = dict().fromkeys(self.state_names,None)
        # interpolate next period's V
        
        
        
        for ist in range(self.num_states):
            is_state = (self.state[:,t]==ist)
            if not np.any(is_state):
                continue
            
            nind = np.sum(is_state)
            ind = np.where(is_state)[0]
            
            for name in self.state_names:
                VV = self.V[t+1][name]['V']
                V_i = VV[self.iassets[ind,t+1],self.iexo[ind,t+1]]
                V_ip = VV[self.iassets[ind,t+1]+1,self.iexo[ind,t+1]]
                V_int = self.wassets[ind,t+1]*V_i + (1-self.wassets[ind,t+1])*V_ip
                Vnext[name] = {'V': V_int,'c': self.V[t+1][name]['c'][self.iassets[ind,t+1],self.iexo[ind,t+1]]}
            
            
            sname = self.state_names[ist]
            
            transition = self.setup.transitions[sname]
            
            
            destinations = transition.elem_probabilities(Vnext)
            
            new_states = np.arange(self.num_states)
            state_probs = np.zeros((nind,self.num_states),np.float32)
            
            for i in new_states:
                if self.state_names[i] in destinations:
                    state_probs[:,i] = destinations[self.state_names[i]]
            
            assert np.all(np.abs(np.sum(state_probs,axis=1) - 1)<1e-5)
                
            # this is probably very slow
            for j in range(nind):
                self.state[ind[j],t+1] = np.random.choice(new_states,size=1,p=state_probs[j,:].squeeze())
            
        
    
    def simulate(self):
        for t in range(self.T-1):
            self.anext(t)
            self.iexonext(t)
            self.state_next(t)