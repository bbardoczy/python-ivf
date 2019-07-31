# these are tools for value functions
import numpy as np
from interp_my import interpolate_vector_matrix
from numba import jit

def agrid_fun(n=400,amax=40,amin=0,pivot=0.25):
    a = np.geomspace(amin+pivot,amax+pivot,n) - pivot
    a[0] = amin
    return a

def zgrid_uniform(z_sd,nz=25,nsd=3):
    if z_sd.size == 1:
        z = np.linspace(-nsd*z_sd,nsd*z_sd,nz)
    else:
        z = list()
        for zsd_i in z_sd:
            z = z + [zgrid_uniform(zsd_i,nz,nsd)]
    return z

#@jit(nopython=True)


def v_optimize(m,agrid,beta,EV,ns=200,minc=0.001,u=None):
    # this solves consumption-savings problem in the form
    # u(c) + beta*EV(m-c).
    # EV is assumed to be integrated, c is in [minc*m,m].
    # it does so by bruteforcing ns possible points for savings such that
    # s = share*m
    
    if u is None:
        u = lambda x : np.log(x)
    
    V = np.empty(m.shape,np.float64)
    c = np.empty(m.shape,np.float64)
    s = np.empty(m.shape,np.float64)
    
    ns_low = int(ns/4)
    sshare = np.concatenate( (np.linspace(0,0.5,ns_low), np.linspace(0.5+(1/ns),1-minc,ns-ns_low)))
    
    for iz in range(0,m.shape[1]):
           
        mi = m[:,iz]
        
        ap_val = np.expand_dims(mi,1) * sshare
        c_val  = np.expand_dims(mi,1) - ap_val
        uc    = u(c_val)
        
        V_opt, i_opt = v_optimize_one(EV[:,iz],agrid,uc,ap_val,beta)
        
        #V[:,iz] = V_val [np.arange(0,agrid.size),i_opt]
        
        V[:,iz] = V_opt
        
        # njit-specific
        for ia in range(0,m.shape[0]):
            c[ia,iz] = c_val [ia,i_opt[ia]]
            s[ia,iz] = ap_val[ia,i_opt[ia]]
            
    return V, c, s
    

@jit(nopython=True)
def v_optimize_one(EV_one,agrid,uc,ap,beta):
        
    nm = uc.shape[0]
    
    EV_val = interpolate_vector_matrix(agrid,EV_one,ap)
    V_val  = uc + beta*EV_val
    
    i_opt = np.empty(nm,np.int64)
    V_opt = np.empty(nm,np.float64)
    
    #i_opt  = np.argmax(V_val, axis=1) # non njit-implementation
    
    for im in range(0,nm):
        io        = np.argmax(V_val[im,:])
        i_opt[im] = io
        V_opt[im] = V_val[im,io]
        
        
    return V_opt, i_opt
    
    

def smooth_max(v0,v1,eps):
    assert np.all(v0.shape == v1.shape)
    v_max = np.maximum(v0,v1)
    v_min = np.minimum(v0,v1)
    
    v_result = v_max + eps * np.log(1+np.exp( (v_min - v_max) / eps))
    
    return v_result

