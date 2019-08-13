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
#from vf_tools import smooth_max, smooth_p0
from numba import njit
from valueFunctions import valuefunction, grid_dim_linear, grid_dim_discrete




def agrid_fun(n=200,amax=20,amin=0,pivot=0.25):
    a = np.geomspace(amin+pivot,amax+pivot,n) - pivot
    a[0] = amin
    return a

def vfint(vfin,Pi):
    # Pi[i,j] is probability to go from i to j
    # Pi must be two dimensional and its rows must add to 1
    assert Pi.ndim == 2
    return np.dot(vfin['V'],Pi.T)



def Vnext_egm(agrid,labor_income,EV_next,EMU_next,Pi,R,beta,m=None,u=None,mu_inv=None,uefun=None):
    
    
    if m is None: # we can override m
        m = np.float64( R*agrid[:,np.newaxis] + labor_income )
        
    
    dc = True
        
        
    
    if (EV_next is None):
        V, c, s = (u(m), m, np.zeros_like(m))
    else:
        
        
        
        c_of_anext = mu_inv( np.maximum(EMU_next,1e-16) )
        m_of_anext = c_of_anext + agrid[:,np.newaxis]
        
        c = np.empty_like(m)
        s = np.empty_like(m)
        V = np.empty_like(m)
        
        uecount = 0
        
        for i in range(m_of_anext.shape[1]):
            
            if not np.all(np.diff(m_of_anext[:,i])>0):
                
                assert dc, "Non-monotonic m with no switching?"
                
                uecount += 1
                # use upper envelope routine
                (c_i, V_i) = (np.empty_like(m_of_anext[:,i]), np.empty_like(m_of_anext[:,i]))
                uefun(agrid,m_of_anext[:,i],c_of_anext[:,i],EV_next[:,i],m[:,i],c_i,V_i)
                    
                c[:,i] = c_i
                V[:,i] = V_i
                s[:,i] = m[:,i] - c[:,i]
                    
                
            else:
                
                # manually re-interpolate w/o upper envelope
                
                a_of_anext_i = (1/R)*( m_of_anext[:,i] - labor_income[i] )                
            
                #anext_of_a[:,i] = interpolate_nostart(a_of_anext[:,i],agrid,agrid)
                anext_of_a_i = np.interp(agrid,a_of_anext_i,agrid)
                EV_next_a_i  = np.interp(anext_of_a_i,agrid,EV_next[:,i])
                    
                anext_of_a_i = np.maximum(anext_of_a_i,0)
                c[:,i] = m[:,i] - anext_of_a_i
                s[:,i] = anext_of_a_i
                V[:,i] = u(c[:,i]) + beta*EV_next_a_i          
            
        
        if uecount>0:
            print("Upper envelope count: {}".format(uecount))
        
    
    return V, c, s
    
def Vnext_vfi(agrid,labor_income,EV_next,c_next,Pi,R,beta,m=None,u=None,mu=None,mu_inv=None,uefun=None):
    
    
    from vf_tools import v_optimize
    
    if m is None: # we can override m
        m = np.float64( R*agrid[:,np.newaxis] + labor_income )
        
    if (EV_next is None):
        V, c, s = (u(m), m, np.zeros_like(m))
    else:
        V, c, s = v_optimize(m,agrid,beta,EV_next,ns=200,u=u)
    return V, c, s


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


def v0(obj):
    
    o = obj.options    
    while not isinstance(o[0],str):
        o = o[0].options
    
    assert type(o[0]) is str
    
    return o[0]

class switch:
    
    @staticmethod
    def v0def(opts):        
          
        while not isinstance(opts[0],str):
            opts = opts[0].options
        
        assert type(opts[0]) is str
        return opts[0]

class choice(switch):
    def __init__(self,options,eps):
        self.options = options
        self.eps = eps        
        self.v0 = self.v0def(options)
        
        assert isinstance(options,list), 'Choices must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in choices!'
    
        
        
class shock(switch):
    def __init__(self,options,ps):
        self.options = options
        self.ps      = ps        
        self.v0 = self.v0def(options)
        
        assert np.abs(sum(ps) - 1)<1e-6
        assert isinstance(options,list), 'Options must be a list!'
        assert all([isinstance(i,(switch,str)) for i in options]), 'Unsupported thing in options!'
        
   
def ev(Vdict,transition,mu):
    
    Vbase = [Vdict[opt]['V'] if (type(opt) is str)
             else ev(Vdict,opt,mu)[0]
             for opt in transition.options]
    
    EMUbase = [mu(Vdict[opt]['c']) if (type(opt) is str)
                else ev(Vdict,opt,mu)[1]
                for opt in transition.options]
    
    Vstart = Vdict[transition.v0]
    
    if len(Vbase)==1:
        return Vstart['V'], mu(Vstart['c'])
    else:
        if type(transition) is choice:
            EV, p = Vstart.combine_endo(Vbase,transition.eps,return_p=True)
            EMU   = Vstart.combine_exo(EMUbase,p[1:])
            return EV, EMU
        elif type(transition) is shock:
            EV = Vstart.combine_exo(Vbase,transition.ps[1:])
            EMU  = Vstart.combine_exo(EMUbase,transition.ps[1:])
            return EV, EMU
    
    

# this runs the file    
if __name__ == "__main__":
    
    
    T = 20
    sigma = 1
    R = 1.03
    beta = 0.97
    
    
    trange = np.arange(0,T)
    trange = trange/np.mean(trange)
    
    a0, a1, a2 = 0.0, 0.05, 0.05
    ztrend = a0 + a1*trange + a2*(trange**2)
    b0 = 0
    b1 = -0.1
    gtrend = np.exp(b0 + b1*trange)
    
    
    agrid = agrid_fun(amax=40)
    
    g_kid = 0.05
    z_kid = 0.2
    u_kid_c = 0.0
    u_kid_add = 0.05
    phi_in = 1
    phi_out = 0.4
    pback = 0.25
    
    eps = 0.00
    
    a = dict(sigma_z_init=0.15,sigma_z=0.1,nz=7,
                 sigma_g=0.1,rho_g=0.8,ng=7,smin=0,smax=4.12,ns=16,T=T,mult=gtrend)
    
    zgs_GridList_nok,   zgs_MatList_nok  = generate_zgs(**a)
    zgs_GridList_k,     zgs_MatList_k  = generate_zgs(**a,fun = lambda g : np.maximum( g - g_kid, 0 ) )
    
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
                    'One child, out': lambda grid, t : phi_out*np.exp(grid[:,0] + grid[:,2] - z_kid  + ztrend[t]).T,
                    'One child, in':  lambda grid, t : phi_in*np.exp(grid[:,0] + grid[:,2] - z_kid  + ztrend[t]).T
                   }
    
    u, mu, mu_inv, ue = define_u(sigma,u_kid_c,u_kid_add)
    
    
        
    
    V = [{ 'No children':None, 'One child, out':None, 'One child, in':None }]*T
    
    descriptions = [*V[0]] 
    
    for desc in descriptions:
        
        Vcs = iterator(agrid,labor_income[desc](zgs_GridList[desc][-1],T-1),None,None,None,R,beta,u=u[desc],mu_inv=mu_inv[desc],uefun=ue[desc])  
        V[T-1][desc] = vpack(Vcs,agrid,zgs_GridList[desc][-1],T-1,desc)
        
        
        
    for t in reversed(range(T-1)):
        
        Vnext = V[t+1]
        Vcurrent = V[t] # passed by reference
        
        for desc in descriptions:   
            
            
            gri, ma = zgs_GridList[desc][t], zgs_MatList[desc][t]            
            
            integrate = lambda V : np.dot(V,ma.T)
            
            if desc == "One child, in":
                transition = choice(["One child, in"],eps)                
            elif desc == "One child, out":                
                transition = shock(["One child, out","One child, in"],[1-pback,pback])
            elif desc == "No children":     
                if_child = shock(["One child, out","One child, in"],[1,0])
                transition = choice(["No children",if_child],eps)              
            else:
                raise Exception("Unsupported type?")
                
                
            Vcomb, MU_comb = ev(Vnext,transition,mu=mu[desc])
                
            assert np.all(MU_comb > 0)            
            
            EV, EMU  = integrate(  Vcomb  ), integrate(  MU_comb  )            
            
            Vcs = iterator(agrid,labor_income[desc](gri,t),EV,EMU,ma,R,beta,u=u[desc],mu_inv=mu_inv[desc],uefun=ue[desc])
            
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
    
    
    
        
    
    