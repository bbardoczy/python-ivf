#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jul 27 13:39:24 2019

@author: Egor Kozlov

This thing runs the main calculations for IVF project. Nothing important
here now

"""

from generate_grid import generate_zgs
import matplotlib.pyplot as plt
import numpy as np
from vf_tools import smooth_max, smooth_p0
import uenv
from numba import njit




def agrid_fun(n=800,amax=20,amin=0,pivot=0.25):
    a = np.geomspace(amin+pivot,amax+pivot,n) - pivot
    a[0] = amin
    return a

def vfint(vfin,Pi):
    # Pi[i,j] is probability to go from i to j
    # Pi must be two dimensional and its rows must add to 1
    assert Pi.ndim == 2
    return np.dot(vfin['V'],Pi.T)



def Vnext_egm(agrid,labor_income,EV_next,c_next,Pi,R,beta,m=None,u=None,mu=None,mu_inv=None):
    
    
    if m is None: # we can override m
        m = np.float64( R*agrid[:,np.newaxis] + labor_income )
        
    
    if type(c_next) is tuple:
        dc = True
        c0 = c_next[0]
        c1 = c_next[1]
        p0 = c_next[2]
    else:
        dc = False
        
        
    
    if (EV_next is None):
        V, c, s = (u(m), m, np.zeros_like(m))
    else:
        
        if dc:
            mu_next = p0*mu(c0) + (1-p0)*mu(c1)
        else:
            mu_next = mu(c_next)
        
        muc = beta*R*np.dot(mu_next,Pi.T)
        
        c_of_anext = mu_inv( np.maximum(muc,1e-16) )
        m_of_anext = c_of_anext + agrid[:,np.newaxis]
        #v_of_anext = u(c_of_anext) + EV_next
        
        
        assert np.all(m_of_anext >= np.min(m,axis=0)), 'fix at 0 is needed'
        
        
        if True:#not np.all(np.diff(m_of_anext,axis=0)>0):
            
            ue = uenv.create(u,False)
            
            c_in = np.empty_like(m)
            v_in = np.empty_like(m)
            
            
            for j in range(c_in.shape[1]):
                ue(agrid,m_of_anext[:,j],c_of_anext[:,j],EV_next[:,j],m[:,j],c_in[:,j],v_in[:,j])
                
            c = c_in
            s = m - c_in
            V = v_in
                
        else:
            c_in = c_of_anext
            m_in = c_in + agrid[:,np.newaxis]
            a_of_anext = (1/R)*( m_in - labor_income )
            anext_of_a = np.zeros_like(a_of_anext)
            EV_next_a  = np.empty_like(a_of_anext)
        
            for i in range(0,a_of_anext.shape[1]):
                #anext_of_a[:,i] = interpolate_nostart(a_of_anext[:,i],agrid,agrid)
                anext_of_a[:,i] = np.interp(agrid,a_of_anext[:,i],agrid)
                EV_next_a[:,i]  = np.interp(anext_of_a[:,i],agrid,EV_next[:,i])
                
            anext_of_a = np.maximum(anext_of_a,0)
            c_of_a = m - anext_of_a
            c, s = (c_of_a, anext_of_a)
            
            
            V = u(c_of_a) + beta*EV_next_a            
            
        # upper envelope should be here
        
        
        
    
    return V, c, s
    
def Vnext_vfi(agrid,labor_income,EV_next,c_next,Pi,R,beta,m=None,u=None,mu=None,mu_inv=None):
    
    
    from vf_tools import v_optimize
    
    if m is None: # we can override m
        m = np.float64( R*agrid[:,np.newaxis] + labor_income )
        
    if (EV_next is None):
        V, c, s = (u(m), m, np.zeros_like(m))
    else:
        V, c, s = v_optimize(m,agrid,beta,EV_next,ns=200,u=u)
    return V, c, s


def vpack(Vcs,agrid,zgsgrid,time,description=""):
    from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete
    
    Vout = valuefunction([grid_dim_linear(agrid),grid_dim_discrete(zgsgrid)],{'V':Vcs[0],'c':Vcs[1],'s':Vcs[2]},time,description)
    return Vout
    

# this runs the file    
if __name__ == "__main__":
    
    
    T = 40
    sigma = 1
    R = 1.03
    beta = 0.97
    
    agrid = agrid_fun(amax=100)
    
    g_kid = 0.0
    z_kid = 0.0
    u_kid_c = 0.0
    u_kid_add = 0.5
    phi_kid = 0.01
    
    eps = 0.01
    
    zgs_GridList = list()
    zgs_MatList = list()
    
    a = dict(sigma_z_init=0.015,sigma_z=0.01,nz=10,
                 sigma_g=0.1,rho_g=0.8,ng=10,smin=0,smax=0.0412,ns=20,T=T)
    
    zgs_GridList_nok,   zgs_MatList_nok  = generate_zgs(**a)
    zgs_GridList_k,     zgs_MatList_k  = generate_zgs(**a,fun = lambda g : np.maximum( g - g_kid, 0 ) )
    
    zgs_GridList = [zgs_GridList_nok, zgs_GridList_k]
    zgs_MatList  = [zgs_MatList_nok,  zgs_MatList_k ]
    
    iterator = Vnext_egm
    
    labor_income = [lambda grid : 10*np.exp(grid[:,0] + grid[:,2]).T, lambda grid : 10*phi_kid*np.exp(grid[:,0] + grid[:,2] - z_kid).T]
    
    uselog = True if sigma == 1 else False
    
    
    
    
    if uselog:        
        @njit
        def u_nok(x):
            return np.log(x)
        
        @njit
        def u_k(x):
            return np.log(x) + u_kid_c + u_kid_add
        
        
        u =       [u_nok, u_k]
        mu =      [lambda x  : 1/x  ]*2
        mu_inv =  [lambda MU : 1/MU ]*2
    else:
        
        
        factor = np.exp(u_kid_c*(1-sigma))
        
        @njit
        def u_nok(x: np.float64) -> np.float64:
            return (x**(1-sigma))/(1-sigma)
        
        @njit
        def u_k(x: np.float64) -> np.float64:
            return factor*(x**(1-sigma))/(1-sigma) + u_kid_add
        
        u =      [u_nok,  u_k]
        mu =     [lambda x  : (x**(-sigma)),             lambda x  : factor*(x**(-sigma))]
        mu_inv = [lambda MU : (MU**(-1/sigma)),          lambda MU : ((MU/factor)**(-1/sigma))]
        
        #u =      [lambda x  : (x**(1-sigma))/(1-sigma),  lambda x  : (x**(1-sigma))/(1-sigma) + u_kid]
        #mu =     [lambda x  : (x**(-sigma)),             lambda x  : (x**(-sigma))]
        #mu_inv = [lambda MU : (MU**(-1/sigma)),          lambda MU : ((MU)**(-1/sigma))]
        
        
    V = list()
    
    desc = ["No children","One child"]
    
    for igrid in [0,1]:
        
        Vcs = iterator(agrid,labor_income[igrid](zgs_GridList[igrid][-1]),None,None,None,R,beta,u=u[igrid],mu=mu[igrid],mu_inv=mu_inv[igrid])  
        #V, c, s = [Vlast], [clast], [slast]
        V.append([vpack(Vcs,agrid,zgs_GridList[igrid][-1],T-1,desc[igrid])])
        
    for t in reversed(range(T-1)):
        for igrid in [0,1]:
            
            
            
            gri = zgs_GridList[igrid][t]
            ma  = zgs_MatList[igrid][t]
            
            if igrid == 1:
                EV = np.dot( V[igrid][0]['V'],  ma.T)
                cnext = V[igrid][0]['c']
            else:
                
                EV = np.dot( smooth_max(V[0][0]['V'],V[1][0]['V'],eps),          ma.T)
                c0 = V[0][0]['c']
                c1 = V[1][0]['c']
                p0 = smooth_p0(V[0][0]['V'],V[1][0]['V'],eps)
                cnext = (c0,c1,p0)
            
            
            
            Vcs = iterator(agrid,labor_income[igrid](gri),EV,cnext,ma,R,beta,u=u[igrid],mu=mu[igrid],mu_inv=mu_inv[igrid])
           
            V[igrid] = [vpack(Vcs,agrid,gri,t,desc[igrid])] + V[igrid]
        
        
    it = 0
    
    print(V[0][0][5,1000])
    print(V[1][0][5,1000]-V[0][0][5,1000])
    
    
    plt.cla()
    #plt.subplot(211)
    V[0][it].plot_value('s')
    V[1][it].plot_value('s')
    #plt.subplot(212)
    #V[0][it].plot_diff(V[1][it],'V')
    plt.legend()
        
    
    