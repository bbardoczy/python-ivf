# this collects iterators for value functions

import numpy as np





def Vnext_egm(agrid,labor_income,EV_next,EMU_next,Pi,R,beta,m=None,u=None,mu_inv=None,uefun=None):
    #raise Exception('This should not work!')
    
    if m is None: # we can override m
        m = np.float64( R*agrid[:,np.newaxis] + labor_income )
        
    
    dc = True
        
        
    
    if (EV_next is None):
        V, c, s = (u(m), m, np.zeros_like(m))
    else:
        
        
        
        c_of_anext_unc = mu_inv( beta*R*np.maximum(EMU_next,1e-16) )
        c_of_anext_ub = m - agrid[-1]
        
        c_of_anext = np.maximum( c_of_anext_unc, c_of_anext_ub)
        
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
                
                
                
                assert np.all(c_of_anext>0)
                
                
                m_i = m[:,i]
                
                assert np.all( np.diff(m_i) > 0)
                
                #m_extra = np.append(m_i,(1.05*m_i[-1],1.1*m_i[-1],1.2*m_i[-1]))
                
                
                (c_i, V_i) = (np.empty_like(m_of_anext[:,i]), np.empty_like(m_of_anext[:,i]))
                uefun(agrid,m_of_anext[:,i],c_of_anext[:,i],beta*EV_next[:,i],m_i,c_i,V_i)
                    
                c[:,i] = c_i#[:m_i.size]
                V[:,i] = V_i#[:m_i.size]
                s[:,i] = m[:,i] - c[:,i]
                
                
                
                # this is debugger
                
                try:
                    assert np.all(c[:,i]>0)
                except:
                    import matplotlib.pyplot as plt
                    
                    
                    
                    #print(c[:,i])
                    
                    from vf_tools import v_optimize
                    Vv, cc, ss = v_optimize(m_i[:,np.newaxis],agrid,beta,EV_next[:,i][:,np.newaxis],ns=200,u=u)                    
                    
                    print(m[:,i])
                    plt.subplot(211)
                    plt.scatter(m_of_anext[:,i],c_of_anext[:,i],label="pre-UE")
                    plt.scatter(m[:,i],c_i,label="post-UE")
                    plt.scatter(m[:,i],cc,label="vfi")
                    plt.subplot(212)
                    plt.scatter(agrid,EV_next[:,i])
                    plt.legend()
                    
                    plt.pause(0.05)
                    #print((m[:,i],m_of_anext[:,i],c_of_anext[:,i],s[:,i]))
                    raise Exception('wow')
                    
                
            else:
                
                # manually re-interpolate w/o upper envelope
                
                a_of_anext_i = (1/R)*( m_of_anext[:,i] - labor_income[i] )                
            
                #anext_of_a[:,i] = interpolate_nostart(a_of_anext[:,i],agrid,agrid)
                anext_of_a_i = np.interp(agrid,a_of_anext_i,agrid)
                anext_of_a_i = np.maximum(anext_of_a_i,0)
                
                
                EV_next_a_i  = np.interp(anext_of_a_i,agrid,EV_next[:,i])
                    
                c[:,i] = m[:,i] - anext_of_a_i
                s[:,i] = anext_of_a_i
                V[:,i] = u(c[:,i]) + beta*EV_next_a_i          
                assert np.all(s[:,i]>=0)
                
        if uecount>0:
            print("Upper envelope count: {}".format(uecount))
        
        assert np.all(s>=0)
    return V, c, s
    
def Vnext_vfi(agrid,labor_income,EV_next,c_next,Pi,R,beta,m=None,u=None,mu=None,mu_inv=None,uefun=None):
    
    
    from vf_tools import v_optimize
    
    if m is None: # we can override m
        m = np.array( R*agrid[:,np.newaxis] + labor_income , np.float64)
        
    if (EV_next is None):
        V, c, s = (u(m), m, np.zeros_like(m))
    else:
        V, c, s = v_optimize(m,agrid,beta,EV_next,ns=200,u=u)
    return V, c, s