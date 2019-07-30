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





def agrid_fun(n=400,amax=40,amin=0,pivot=0.25):
    a = np.geomspace(amin+pivot,amax+pivot,n) - pivot
    a[0] = amin
    return a

def vfint(vfin,Pi):
    # Pi[i,j] is probability to go from i to j
    # Pi must be two dimensional and its rows must add to 1
    assert Pi.ndim == 2
    return np.dot(vfin['V'],Pi.T)

def Vnext_egm(agrid,zgsgrid,EV_next,c_next,Pi,R,beta,sigma,m=None):
    
    u = lambda c, e : ( c**(1-sigma) ) / (1-sigma)
    
    if m is None: # we can override m
        m = np.float64( R*agrid[:,np.newaxis] + np.exp(zgsgrid[:,0].T + zgsgrid[:,2].T) )
    
    if (EV_next is None):
        V, c, s = (u(m,zgsgrid), m, np.zeros_like(m))
    else:
        
        muc = beta*R*np.dot(c_next**(-sigma),Pi.T)
        c_of_anext = ( np.maximum(muc,1e-15) )**(-1/sigma)
        a_of_anext = (1/R)*( c_of_anext + agrid[:,np.newaxis] - np.exp(zgsgrid[:,0].T + zgsgrid[:,2].T) )
        anext_of_a = np.zeros_like(a_of_anext)
        
        for i in range(0,a_of_anext.shape[1]):
            #anext_of_a[:,i] = interpolate_nostart(a_of_anext[:,i],agrid,agrid)
            anext_of_a[:,i] = np.interp(agrid,a_of_anext[:,i],agrid)
            
        anext_of_a = np.maximum(anext_of_a,0)
        c_of_a = m - anext_of_a
        c, s = (c_of_a, anext_of_a)
        V = u(c_of_a,zgsgrid) + beta*EV_next
    
    return V, c, s
    
def Vnext_vfi(agrid,zgsgrid,EV_next,c_next,Pi,R,beta,sigma,m=None):
    
    
    from vf_tools import v_optimize
    
    u = lambda c, e : ( c**(1-sigma) ) / (1-sigma)
    
    if m is None: # we can override m
        m = np.float64( R*agrid[:,np.newaxis] + np.exp(zgsgrid[:,0].T + zgsgrid[:,2].T) )
        
    if (EV_next is None):
        V, c, s = (u(m,zgsgrid), m, np.zeros_like(m))
    else:
        V, c, s = v_optimize(m,agrid,zgsgrid,sigma,beta,EV_next,ns=200,u=u)
    return V, c, s


def vpack(Vcs,agrid,zgsgrid,time,description=""):
    from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete
    
    Vout = valuefunction([grid_dim_linear(agrid),grid_dim_discrete(zgsgrid)],{'V':Vcs[0],'c':Vcs[1],'s':Vcs[2]},time,description)
    return Vout
    


# this runs the file    
if __name__ == "__main__":
    zgs_GridList, zgs_MatList = generate_zgs()
    
    T = 20
    sigma = 2
    R = 1.02
    beta = 0.99
    
    agrid = agrid_fun()
    
    
    iterator = Vnext_vfi
    
    Vcs = iterator(agrid,zgs_GridList[-1],None,None,None,R,beta,sigma,None)
    
    #V, c, s = [Vlast], [clast], [slast]
    V = [vpack(Vcs,agrid,zgs_GridList[-1],T-1,"individual")]
    
    for t in reversed(range(T-1)):
        EV = np.dot(V[0]['V'],zgs_MatList[t].T)
        Vcs = iterator(agrid,zgs_GridList[t],EV,V[0]['c'],zgs_MatList[t],R,beta,sigma)
        V = [vpack(Vcs,agrid,zgs_GridList[t],t,"individual")] + V
        
        
    
        
    
    plt.cla()
    #plt.subplot(211)
    V[5].plot_value('V')
    V[10].plot_value('V')
    plt.legend()
        
    
    