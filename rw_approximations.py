# this contains different appriximations for random walk
# this also contains few auxiliary functions like truncated normal

# Important note: if V is [na,nz] array, to get expected value of V 
# you need to multiply it by transpose of Pi produced by these routines:
# EV = V*Pi.transpose(). Failure to do so produces wrong numbers w/o error


def sd_rw(T,sigma_persistent,sigma_init):
    import numpy as np
    return np.sqrt(sigma_init**2 + np.arange(0,T)*(sigma_persistent**2))
    
def sd_rw_trans(T,sigma_persistent,sigma_init,sigma_transitory):
    return sd_rw(T, sigma_persistent, sigma_init)

    
    
def normcdf_tr(z,nsd=5):
        import numpy as np
        from scipy import stats as st
        
        z = np.minimum(z, nsd*np.ones_like(z))
        z = np.maximum(z,-nsd*np.ones_like(z))
            
        pup = st.norm.cdf(nsd)
        pdown = st.norm.cdf(-nsd)
        const = pup - pdown
        
        return (st.norm.cdf(z)-pdown)/const
        
def tauchen_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=50,nsd=3):
    import numpy as np

    # start with creating list of points
    sd_z = sd_rw(T,sigma_persistent,sigma_init)
    X = list()
    Pi = list()

    for t in range(0,T):
        X = X + [np.linspace(-nsd*sd_z[t],nsd*sd_z[t],num=npts)]

    # then define a list transition matrices
    # note that Pi[t] is transition matrix from X[t] to X[t+1]

    for t in range(1,T):
        h = 2*nsd*sd_z[t]/(npts-1) # step size
        Pi_here = np.zeros([npts,npts])

        Pi_here[:,0] = normcdf_tr( (X[t][0] - X[t-1][:] + h/2) / sigma_persistent)
        Pi_here[:,-1] = 1 - normcdf_tr( (X[t][-1] - X[t-1][:] - h/2) / sigma_persistent)

        for i in range(1,npts-1):
            Pi_here[:,i] = normcdf_tr( (X[t][i] - X[t-1][:] + h/2) / sigma_persistent) - normcdf_tr( (X[t][i] - X[t-1][:] - h/2) / sigma_persistent)

        #print(Pi_here)
        #print(np.sum(Pi_here,axis=1))
        #assert(np.all( abs( np.sum(Pi_here,axis=1) -np.ones(npts) ) < 1e-6 ))

        Pi = Pi + [Pi_here]
        
        
    Pi = Pi + [None] # last matrix is not defined

    return X, Pi


def rouw_nonst(T=40,sigma_persistent=0.05,sigma_init=0.2,npts=10):
    import numpy as np
    sd_z = sd_rw(T,sigma_persistent,sigma_init)
    assert(npts>=2)
    Pi = list()
    X = list()
    
    for t in range(0,T):
        nsd = np.sqrt(npts-1)
        X = X + [np.linspace(-nsd*sd_z[t],nsd*sd_z[t],num=npts)]
        if t >= 1: Pi = Pi + [rouw_nonst_one(sd_z[t-1],sd_z[t],npts)]
            
    Pi = Pi + [None] # last matrix is not defined
    
    return X, Pi

def rouw_st(sigma=0.05,rho=0.8,npts=10):
    import numpy as np
    sd_z = sigma/np.sqrt(1-rho**2)
    assert(npts>=2)
    Pi = list()
    
    nsd = np.sqrt(npts-1)
    X = np.linspace(-nsd*sd_z,nsd*sd_z,num=npts)
            
    pi0 = 0.5*(1+rho)
    Pi = np.array([[pi0,1-pi0],[1-pi0,pi0]])
    assert(pi0<1)
    assert(pi0>0)
    for n in range(3,npts+1):
        A = np.zeros([n,n])
        A[0:(n-1),0:(n-1)] = Pi
        B = np.zeros([n,n])
        B[0:(n-1),1:n] = Pi
        C = np.zeros([n,n])
        C[1:n,1:n] = Pi
        D = np.zeros([n,n])
        D[1:n,0:(n-1)] = Pi
        Pi = pi0*A + (1-pi0)*B + pi0*C + (1-pi0)*D
        Pi[1:n-1] = 0.5*Pi[1:n-1]
        
        assert(np.all(np.abs(np.sum(Pi,axis=1)-1)<1e-5 ))
    
    return X, Pi

def rouw_nonst_one(sd0,sd1,npts):
    import numpy as np
    # this generates one-period Rouwenhorst transition matrix
    assert(npts>=2)
    pi0 = 0.5*(1+(sd0/sd1))
    Pi = np.array([[pi0,1-pi0],[1-pi0,pi0]])
    assert(pi0<1)
    assert(pi0>0)
    for n in range(3,npts+1):
        A = np.zeros([n,n])
        A[0:(n-1),0:(n-1)] = Pi
        B = np.zeros([n,n])
        B[0:(n-1),1:n] = Pi
        C = np.zeros([n,n])
        C[1:n,1:n] = Pi
        D = np.zeros([n,n])
        D[1:n,0:(n-1)] = Pi
        Pi = pi0*A + (1-pi0)*B + pi0*C + (1-pi0)*D
        Pi[1:n-1] = 0.5*Pi[1:n-1]
        
        assert(np.all(np.abs(np.sum(Pi,axis=1)-1)<1e-5 ))
    
    return Pi