#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main file

"""


import numpy as np
from between_states import ev_emu, at_iw
from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete
from grids_and_functions import setupClass
from vf_iterators import Vnext_egm, Vnext_vfi
from interp_my import interpolate_nostart
from mc_tools import mc_simulate
from numba import jit

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
        if name == "EGM" or name=="EGM-verbose":
            
            verbose = (name=="EGM-verbose")
            
            s = self.setup
            
            def iterate(desc,t,EV,EMU):
                gri, ma = s.zgs_Grids[desc][t], s.zgs_Mats[desc][t]     
                Vcs = Vnext_egm(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])                
                
                if verbose:
                    print('Solved {} at t = {}'.format(desc,t))
                
                return vpack(Vcs,t,desc)
        
            def initialize(desc,t):
                return iterate(desc,t,None,None)
            
        elif name == "EGM-debug":
            
            
            s = self.setup
            
            def iterate(desc,t,EV,EMU):
                gri, ma = s.zgs_Grids[desc][t], s.zgs_Mats[desc][t]   
                try:
                    Vcs = Vnext_egm(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])                
                except:
                    print('warning: egm fails')
                    Vcs = Vnext_vfi(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])
                
                return vpack(Vcs,t,desc)
        
            def initialize(desc,t):
                return iterate(desc,t,None,None)
            
        
        elif name == "EGM-compare":
            
            s = self.setup
            
            def iterate(desc,t,EV,EMU):
                gri, ma = s.zgs_Grids[desc][t], s.zgs_Mats[desc][t]     
                Vcs  = Vnext_egm(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc])                
                Vcs2 = Vnext_vfi(s.agrid,s.labor_income[desc](gri,t),EV,EMU,ma,s.pars['R'],s.pars['beta'],u=s.u[desc],mu_inv=s.mu_inv[desc],uefun=s.ue[desc]) 
                
                print('Maximum differences:')
                
                g = lambda x : np.log(1+x)
                
                mdiff_v = np.max(np.abs(Vcs2[0] - Vcs[0]))
                mdiff_c = np.median(np.abs(g(Vcs2[1]) - g(Vcs[1])))
                mdiff_s = np.median(np.abs(g(Vcs2[2]) - g(Vcs[2])))
                
                print( 'V (max): {}, c (median): {}, s (median): {}'.format(mdiff_v,mdiff_c,mdiff_s) )
                
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
        #self.descriptions = ['No children, fertile', 'One child, out, fertile', 'One child, in, fertile','Two children, out, fertile', 'Two children, in, fertile']#[*self.V[0]]
        
        self.descriptions = list(self.setup.u.keys())
        
        T = self.setup.pars['Tdie']
        
        
        
        integrate = lambda V, mat : V if mat is None else np.dot(V,mat.T)   
        
        # this part is backward iteration
        for t in reversed(range(T)):
            
            Vthis = list()
            
            s = self.setup
            
            for desc in self.descriptions:
            
                if t < s.time_limits[desc][0] or t >= s.time_limits[desc][1]:
                    Vthis.append(None)
                    continue
                
                if t == T-1:
                    Vthis.append(self.initialize(desc,t))
                else:
                    
                    Vnext = self.V[0]
                    
                    
                    def get_V(desc):
                        
                        ma = s.zgs_Mats[desc][t]          
                        trns = s.transitions_t[t][desc]
                        
                        Vcomb, MU_comb = ev_emu(Vnext,trns,mu=s.mu[desc])                
                        EV, EMU  = integrate(  Vcomb , ma ), integrate(  MU_comb , ma )  
                        VV = self.iterate(desc,t,EV,EMU)
                        return VV
                
                    Vthis.append(get_V(desc))
                
                
            V_dict = dict(zip(self.descriptions,Vthis))
            self.V = [V_dict] + self.V
                #self.V[t][desc] = self.vpack(Vcs,t,desc)
            
    
class Agents:
    def __init__(self,M,N=1000,T=None):
        if T is None:
            T = M.setup.pars['Tinf']
            
        np.random.seed(18)    
        
        self.iassets = np.zeros((N,T),np.int32)
        
        self.wassets = np.ones((N,T),np.float32)
        self.assets = np.empty((N,T))
        self.iexo = np.zeros((N,T),np.int32)
        self.iexo[:,0] = np.random.randint(0,1000,size=N)
        #self.iassets[:,0] = np.random.randint(0,100,size=N)
        
        self.state = np.ones((N,T),np.int32)
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
                self.assets[ind,t+1] = anext_val
                self.iassets[ind,t+1], self.wassets[ind,t+1], atest = interpolate_nostart(agrid,anext_val,agrid,xd_ordered=False)
            except:
                print(t,agrid.shape,anext_val.shape, w_i, s_i, s_ip)
                raise Exception('interpolator stopped working again')
                
            assert np.all(np.abs(atest - anext_val) < 1e-2)
    
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
        
        # interpolate next period's V
        # hese some V may be none if states are not present
        # this is ok
        
        
        
        agrid = self.setup.agrid
        
        
        for ist in range(self.num_states):
            is_state = (self.state[:,t]==ist)
            if not np.any(is_state):
                continue
            
            Vnext = dict().fromkeys(self.state_names,None)
        
            nind = np.sum(is_state)
            ind = np.where(is_state)[0]
            
            sname = self.state_names[ist]
            
            transition = self.setup.transitions_t[t][sname]
            
            
            for name in transition.outcomes:
                if self.V[t+1][name] is None:
                    continue
                VV = self.V[t+1][name]['V']
                V_int = VV[:,self.iexo[ind,t+1]] #at_iw(VV[:,self.iexo[ind,t+1]],self.iassets[ind,t+1],self.wassets[ind,t+1])
                #assert (V_int.shape == VV[:,self.iexo[ind,t+1]].shape)
                # this collects V for ALL asset levels
                # this is required so we can apply offset
                assert V_int.ndim==2
                Vnext[name] = {'V': V_int,'c': np.abs(V_int)}
            
            
            
            destinations = transition.elem_probabilities(Vnext)
            
            for d in destinations.keys():
                destinations[d] = np.diag(at_iw(destinations[d], self.iassets[ind,t+1],self.wassets[ind,t+1]))
                assert destinations[d].size == nind
            
            
            
            
            
            # inside elem_probabilities there is ev_emu
            
            new_states = np.arange(self.num_states)
            state_probs = np.zeros((nind,self.num_states),np.float32)
            
            for i in new_states:
                if self.state_names[i] in destinations:
                    state_probs[:,i] = destinations[self.state_names[i]]
            
            assert np.all(np.abs(np.sum(state_probs,axis=1) - 1)<1e-5)
                
            # this is probably very slow
            
            offsets = np.zeros(nind)
            for j in range(nind):
                self.state[ind[j],t+1] = np.random.choice(new_states,size=1,p=state_probs[j,:].squeeze())
                offsets[j] = transition.offset_prices[self.state_names[self.state[ind[j],t+1]]]
                
            if not np.any(offsets>0): continue
        
            i_change = ind[np.where(offsets>0)]
            
            
            self.assets[i_change,t+1] = self.assets[i_change,t+1] - offsets[np.where(offsets>0)]
            self.iassets[i_change,t+1], self.wassets[i_change,t+1], atest = interpolate_nostart(agrid,self.assets[i_change,t+1],agrid,xd_ordered=False)
            
            assert np.all(self.assets[ind,t+1] >= -1e-3)
            # now let's adjust a
    '''        
    def iwnext(self,t):
        # this converts asset values to indices
        agrid = self.setup.agrid
        
        self.iassets[:,t+1], self.wassets[:,t+1], atest = interpolate_nostart(agrid,self.assets[:,t+1],agrid,xd_ordered=False)
        assert np.all(np.abs(atest - self.assets[:,t+1]) < 1e-2)
    '''
    
    def simulate(self):
        for t in range(self.T-1):
            self.anext(t)
            self.iexonext(t)
            self.state_next(t)
            
            
            