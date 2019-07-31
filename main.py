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



def Vnext_egm(agrid,labor_income,EV_next,c_next,Pi,R,beta,m=None,u=None,mu=None,mu_inv=None):
    
    
    if m is None: # we can override m
        m = np.float64( R*agrid[:,np.newaxis] + labor_income )
        
    
    if (EV_next is None):
        V, c, s = (u(m), m, np.zeros_like(m))
    else:
        
        muc = beta*R*np.dot(mu(c_next),Pi.T)
        c_of_anext = mu_inv( np.maximum(muc,1e-16) )
        a_of_anext = (1/R)*( c_of_anext + agrid[:,np.newaxis] - labor_income )
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
    zgs_GridList, zgs_MatList = generate_zgs()
    
    T = 20
    sigma = 2
    R = 1.02
    beta = 0.99
    
    agrid = agrid_fun()
    

    
    iterator = Vnext_egm
    
    labor_income = lambda grid : np.exp(grid[:,0] + grid[:,2]).T
    
    
    uselog = True
    
    if uselog:
        u = lambda x : np.log(x)
        mu = lambda x : 1/x
        mu_inv = lambda MU: 1/MU
    else:
        u = lambda x : (x**(1-sigma))/(1-sigma)
        mu = lambda x : (x**(-sigma))
        mu_inv = lambda MU : (MU**(-1/sigma))
    
    Vcs = iterator(agrid,labor_income(zgs_GridList[-1]),None,None,None,R,beta,u=u,mu=mu,mu_inv=mu_inv)
    
    #V, c, s = [Vlast], [clast], [slast]
    V = [vpack(Vcs,agrid,zgs_GridList[-1],T-1,"individual")]
    
    for t in reversed(range(T-1)):
        EV = np.dot(V[0]['V'],zgs_MatList[t].T)
        
        Vcs = iterator(agrid,labor_income(zgs_GridList[t]),EV,V[0]['c'],zgs_MatList[t],R,beta,u=u,mu=mu,mu_inv=mu_inv)
        V = [vpack(Vcs,agrid,zgs_GridList[t],t,"individual")] + V
        
        
    
    plt.cla()
    #plt.subplot(211)
    V[5].plot_value('V')
    V[10].plot_value('V')
    plt.legend()
        
    
    