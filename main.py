#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 27 13:39:24 2019

@author: Egor Kozlov

This thing runs the main calculations for IVF project. Nothing important
here now

"""

from generate_grid import generate_zgs, generate_agrid
import matplotlib.pyplot as plt
import numpy as np
#from vf_tools import smooth_max, smooth_p0
from numba import njit
from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete
from vf_iterators import Vnext_egm
from between_states import ev_emu, shock, choice


def vpack(Vcs,agrid,zgsgrid,time,description=""):
    
    Vout = valuefunction([grid_dim_linear(agrid),grid_dim_discrete(zgsgrid)],{'V':Vcs[0],'c':Vcs[1],'s':Vcs[2]},time,description)
    return Vout
    
def define_u(sigma,u_kid_c,u_kid_add):
    import uenv

    uselog = True if sigma == 1 else False
    
    if uselog:        
        factor = 1
        @njit
        def u_nok(x):
            return np.log(x)        
        @njit
        def u_k(x):
            return np.log(x) + u_kid_c + u_kid_add
        
    else:
        factor = np.exp(u_kid_c*(1-sigma))        
        @njit
        def u_nok(x: np.float64) -> np.float64:
            return (x**(1-sigma))/(1-sigma)        
        @njit
        def u_k(x: np.float64) -> np.float64:
            return factor*(x**(1-sigma))/(1-sigma) + u_kid_add
        
    
    u =  {
            'No children':    u_nok,
            'One child, out': u_k,
            'One child, in':  u_k
         }
    
    mu = {
            'No children':    lambda x  : (x**(-sigma)),
            'One child, out': lambda x  : factor*(x**(-sigma)),
            'One child, in':  lambda x  : factor*(x**(-sigma))
         }
    
    mu_inv = {'No children':    lambda MU : (MU**(-1/sigma)),  
              'One child, out': lambda MU : ((MU/factor)**(-1/sigma)),
              'One child, in':  lambda MU : ((MU/factor)**(-1/sigma))
             }
        
    
    ue = {
            'No children':    uenv.create(u['No children'],False),
            'One child, out': uenv.create(u['One child, out'],False),
            'One child, in':  uenv.create(u['One child, in'],False)
         }
    
    return u, mu, mu_inv, ue


    
# this runs the file    
if __name__ == "__main__":
    
    
    
    pars = dict(T = 20,
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
    
    T = pars['T']
    trange = np.arange(0,pars['T'])
    trange = trange/np.mean(trange)
    
    ztrend = pars['a_z0'] + pars['a_z1']*trange + pars['a_z2']*(trange**2)
    
    gtrend = np.exp(pars['a_g0']+ pars['a_g1']*trange)
    
    agrid = generate_agrid(n = pars['na'], amax=pars['amax'])  
    
    
    
    a = dict(sigma_z_init=pars['sigma_z_init'],sigma_z=pars['sigma_z'],nz=pars['nz'],
                 sigma_g=pars['sigma_g'],rho_g=pars['rho_g'],ng=pars['ng'],smin=0,smax=pars['smax'],ns=pars['ns'],
                 T=pars['T'],mult=gtrend)
    
    zgs_GridList_nok,   zgs_MatList_nok  = generate_zgs(**a)
    zgs_GridList_k,     zgs_MatList_k  = generate_zgs(**a,fun = lambda g : np.maximum( g - pars['g_kid'], 0 ) )
    
    zgs_GridList = {
                    'No children': zgs_GridList_nok,
                    'One child, out': zgs_GridList_k,
                    'One child, in': zgs_GridList_k
                   }
    
    zgs_MatList  = {
                    'No children': zgs_MatList_nok,
                    'One child, out': zgs_MatList_k,
                    'One child, in': zgs_MatList_k
                   }
    
    iterator = Vnext_egm
    
    labor_income = {
                    'No children':    lambda grid, t : np.exp(grid[:,0] + grid[:,2] + ztrend[t]).T,
                    'One child, out': lambda grid, t : pars['phi_out']*np.exp(grid[:,0] + grid[:,2] - pars['z_kid']  + ztrend[t]).T,
                    'One child, in':  lambda grid, t : pars['phi_in']*np.exp(grid[:,0] + grid[:,2]  - pars['z_kid']  + ztrend[t]).T
                   }
    
    u, mu, mu_inv, ue = define_u(pars['sigma'],pars['u_kid_c'],pars['u_kid_add'])
    
    
    V = [{ 'No children':None, 'One child, out':None, 'One child, in':None }]*pars['T']
    
    descriptions = [*V[0]] 
    
    for desc in descriptions:
        
        Vcs = iterator(agrid,labor_income[desc](zgs_GridList[desc][-1],T-1),None,None,None,pars['R'],
                       pars['beta'],u=u[desc],mu_inv=mu_inv[desc],uefun=ue[desc])  
        V[T-1][desc] = vpack(Vcs,agrid,zgs_GridList[desc][-1],T-1,desc)
        
    for t in reversed(range(T-1)):
        
        Vnext = V[t+1]
        Vcurrent = V[t] # passed by reference
        
        for desc in descriptions:   
            
            
            gri, ma = zgs_GridList[desc][t], zgs_MatList[desc][t]            
            
            integrate = lambda V : np.dot(V,ma.T)
            
            if desc == "One child, in":
                transition = choice(["One child, in"],pars['eps'])                
            elif desc == "One child, out":                
                transition = shock(["One child, out","One child, in"],[1-pars['pback'],pars['pback']])
            elif desc == "No children":     
                if_child = shock(["One child, out","One child, in"],[1,0])
                transition = choice(["No children",if_child],pars['eps'])              
            else:
                raise Exception("Unsupported type?")
                
                
            Vcomb, MU_comb = ev_emu(Vnext,transition,mu=mu[desc])
                
            assert np.all(MU_comb > 0)            
            
            EV, EMU  = integrate(  Vcomb  ), integrate(  MU_comb  )            
            
            Vcs = iterator(agrid,labor_income[desc](gri,t),EV,EMU,ma,pars['R'],pars['beta'],u=u[desc],mu_inv=mu_inv[desc],uefun=ue[desc])
            
            Vcurrent[desc] = vpack(Vcs,agrid,gri,t,desc)
        
    
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
    
    
    
        
    
    