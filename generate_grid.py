#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 09:47:32 2019


These functions should generate joint transition probabilities for z, g and s
The last function generate_zgs is aimed to be used externally


@author: egorkozlov
"""


import numpy as np
from trans_unif import transition_uniform
from rw_approximations import rouw_st, rouw_nonst
from mc_tools import trim_matrix, combine_matrices_dependent, combine_matrices_list, combine_matrices_two_lists


def generate_GS_matrix(sigma_g=0.1,rho_g=0.8,ng=10,smin=0,smax=4.12,ns=20,fun = lambda g : np.maximum(g,0),mult_i=1.0):
    # this generates joint transition matrix for carreer growth rate g and
    # skill level s. fun specifies how g maps into changes in s. mult_i is an
    # additional multiplier
    
    trim_level = 0.001
    # approximate g
    G, Pi_G = rouw_st(sigma=sigma_g,rho=rho_g,npts=ng)
    Pi_G = trim_matrix(Pi_G,level=trim_level)
    
    # uniform spacing in S
    sgrid = np.linspace(smin,smax,num=ns)

    # define S transition matrix for each g
    Slist = list()
    for ig in range(G.size):
        snext = np.minimum( np.maximum( sgrid + mult_i*fun(G[ig]), sgrid[0] ), sgrid[-1] )
        assert mult_i >= 0
        
        T = np.zeros((sgrid.size,sgrid.size))
        j_to, p_to = transition_uniform(sgrid,snext)
    
        
        
        for jj in range(sgrid.size):
            T[jj,j_to[jj]]   =      p_to[jj]
            T[jj,j_to[jj]+1] =  1 - p_to[jj]
            
            
        assert (np.all(np.abs(np.sum(T,axis=1)-1)<1e-5 ))
        
        Slist = Slist + [T]
    
    GSgrid, GSMat = combine_matrices_dependent(sgrid,Slist,G,Pi_G)
    
    
    ns = sgrid.size
    
    assert Pi_G[5,4]*Slist[5][3,4] == GSMat[5*ns+3,4*ns+4]
    
    return GSgrid, GSMat


def generate_GS_matrices(sigma_g=0.1,rho_g=0.8,ng=10,smin=0,smax=4.12,ns=20,fun = lambda g : np.maximum(g,0),mult=None):
    if mult is None:
        gs, mat = generate_GS_matrix(sigma_g,rho_g,ng,smin,smax,ns,fun,mult_i=1.0)
    else:
        gs, mat = list(), list()
        for t in range(mult.size):
            gs_i, mat_i = generate_GS_matrix(sigma_g,rho_g,ng,smin,smax,ns,fun,mult_i=mult[t])
            gs = gs + [gs_i]
            mat = mat + [mat_i]        
    return gs, mat
        

def generate_Z_matrices(sigma_z_init=0.15,sigma_z=0.1,nz=10,T=40):
    trim_level = 0.001
    # this generates transition matrices for z
    # z is assumed to be random walk, so this uses 
    # non-stationary method implemented in rouw_nonst
    zval, zMat = rouw_nonst(T=T,sigma_persistent=sigma_z,sigma_init=sigma_z_init)
    zMat = [trim_matrix(z,level=trim_level) for z in zMat if z is not None]
    zMat = zMat + [None]
    return zval, zMat


def generate_zgs(sigma_z_init=0.15,sigma_z=0.1,nz=10,T=40,
                 sigma_g=0.1,rho_g=0.8,ng=10,smin=0,smax=4.12,ns=20,
                 fun = lambda g : np.maximum(g,0), mult=None):
    
    
    zval, zMat = generate_Z_matrices(sigma_z_init,sigma_z,nz,T)
    
    if mult is None:
        gsgrid, mat = generate_GS_matrix(sigma_g,rho_g,ng,smin,smax,ns,fun) 
        zgs_GridList, zgs_MatList = combine_matrices_list(zval,gsgrid,zMat,mat,trim=False)
    else:
        gslist, matlist = generate_GS_matrices(sigma_g,rho_g,ng,smin,smax,ns,fun,mult)
        zgs_GridList, zgs_MatList = combine_matrices_two_lists(zval,gslist,zMat,matlist,trim=False)
    
    return zgs_GridList, zgs_MatList



def generate_agrid(n=200,amax=20,amin=0,pivot=0.25):
    a = np.geomspace(amin+pivot,amax+pivot,n) - pivot
    a[0] = amin
    return a
