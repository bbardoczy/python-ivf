# these are Markov chain tools
# note that this version does not utilize grid_combined class
import numpy as np
from numba import jit

def mc_simulate(statein,Piin,shocks=None):
    # this simulates transition one period ahead for a Markov chain
    import numpy as np
    assert(np.max(statein) < Piin.shape[1] )
    n_in = len(statein)
    Picum = np.cumsum(Piin,axis=1)
    Pi_state = Picum[statein,:]
    #print(Pi_state)
    
    if shocks is None:
        rand = np.random.rand(n_in)[:,np.newaxis]
    else:
        rand = shocks[:,np.newaxis]
    
    rand_state = rand.repeat(Piin.shape[1],axis=1)
    stateout = np.sum((rand_state >= Pi_state),axis=1) # note that numeration starts with 0
    assert(np.max(stateout) < Piin.shape[1])
    return stateout 

# yet another    
    
def mc_init_normal(sigma,X,N=1000,shocks=None):
    # this generates N draws from N(0,sigma^2)
    # then it looks at points at X that are the closest to 
    # these draws points and for each draw it returns its index
    import numpy as np
    if shocks is None:
        ran = sigma*np.random.normal(np.zeros(N))
    else:
        ran = sigma*shocks
    
    #X = x.reshape([x.size,1]).repeat(y.size,axis=1)
    #Y = y.reshape([1,y.size]).repeat(x.size,axis=0)
    
    
    return abs(ran[:,np.newaxis]-X).argmin(axis=1)

    

def trim_matrix(M,level=0.001):
    # this eliminates transition probabilities that are lower than level, 
    # renormalizing remaining probabilities to add up to one
    
    if isinstance(M,list):
        Mout = list()
        for m in M:
            Mout = Mout + [trim_one_matrix(m,level)]
    else:
        Mout = trim_one_matrix(M,level)
     
    return Mout

@jit
def trim_one_matrix(M,level=0.001):
    
    Mout = M
    Mout[np.where(M<level)] = 0
    Mout = Mout / np.sum(Mout,axis=1)[:,np.newaxis]
    return Mout

def combine_matrices(a,b,Pia,Pib,check=True,trim=True,trim_level=0.001):
    # this combines INDEPENDENT transition matrices Pia and Pib
    grid = mat_combine(a,b)
    
    Pi = np.kron(Pia,Pib) if ((Pia is not None) and (Pib is not None)) else None
    
    if Pi is not None:
        if check:
            assert(all(abs(np.sum(Pi,axis=1)-1)<1e-5))
        
        if trim:
            Pi = trim_matrix(Pi,trim_level)
    
    return grid, Pi
  
@jit
def combine_matrices_list(alist,b,Pialist,Pib,check=True,trim=True,trim_level=0.001):
    # this combines each element of Pialist and Pib
    # they assumed to be independent (i.e. Pialist and Pib can be combined in any order)
    grid, Pi = (list(), list())
        
    
    for i in range(0,len(Pialist)):
        gr_a, Pi_a = combine_matrices(alist[i],b,Pialist[i],Pib,check,trim,trim_level)
        grid = grid + [gr_a]
        Pi   = Pi + [Pi_a]
        
    if len(alist) > len(Pialist): # fix in case alist has one more element
        grid = grid + [mat_combine(alist[-1],b)]
        
    return grid, Pi


@jit
def combine_matrices_two_lists(alist,blist,Pialist,Piblist,check=True,trim=True,trim_level=0.001):
    # this combines each element of Pialist and Pib
    # they assumed to be independent (i.e. Pialist and Pib can be combined in any order)
    grid, Pi = (list(), list())
        
    assert len(alist) == len(blist)
    
    
    for i in range(0,len(Pialist)):
        gr_a, Pi_a = combine_matrices(alist[i],blist[i],Pialist[i],Piblist[i],check,trim,trim_level)
        grid = grid + [gr_a]
        Pi   = Pi + [Pi_a]
        
    if len(alist) > len(Pialist): # fix in case alist has one more element
        grid = grid + [mat_combine(alist[-1],blist[-1])]
        
    return grid, Pi
        
def combine_matrices_dependent(a,Pialist,b,Pib):
    # this assumes that there is unique transition matrix of a
    # corresponding to each value of b (so alist[j] is conditional transition
    # matrix of a given b = b[j])
    
    assert len(Pialist) == b.size, "Values in list should correspond to values in b"
    
    grid = mat_combine(b,a) # note the order
    
    n_a = a.size
    n_b = b.size
    
    n_ab = a.size*b.size
    
    
    Pi = np.zeros((n_ab,n_ab))
    
    for jb_from in range(n_b):
        for jb_to in range(n_b):
            Pi[jb_from*n_a:(jb_from+1)*n_a, jb_to*n_a:(jb_to+1)*n_a] = Pib[jb_from,jb_to]*Pialist[jb_from]
        
    
    assert(all(abs(np.sum(Pi,axis=1)-1)<1e-5))
    
    return grid, Pi
    



def mat_combine(a,b):
    # this gets combinations of elements of a and b
    
    a = a[:,np.newaxis] if a.ndim == 1 else a
    b = b[:,np.newaxis] if b.ndim == 1 else b
    
    assert a.ndim==2 and b.ndim==2
    
    l_a = a.shape[0]
    l_b = b.shape[0]
    
    w_a = a.shape[1]
    w_b = b.shape[1]
    
    grid = np.empty((l_a*l_b, w_a+w_b))
    
    for ia in range(l_a):
        grid[ia*l_b:(ia+1)*l_b,0:w_a] = a[ia,:] # this is broadcasting
        grid[ia*l_b:(ia+1)*l_b,w_a:(w_a+w_b)] = b
        
        
    return grid
    
def vec_combine(a,b):
    a_rep = a[:,np.newaxis].repeat(b.size,axis=1)
    b_rep = b[np.newaxis,:].repeat(a.size,axis=0)    
    grid = np.concatenate((a_rep.flatten()[:,np.newaxis],b_rep.flatten()[:,np.newaxis]),axis=1)
    return grid



def ind_combine(ia,ib,na,nb):
    return ia*nb + ib
    
    
    

# this tests combining things
if __name__ == "__main__":


    v0 = np.array([1,2,4])
    v1 = np.array([-1,-2])
    
    v_comb = np.array([[1,-1],[1,-2],[2,-1],[2,-2],[4,-1],[4,-2]])

    assert np.all(mat_combine(v0,v1) == v_comb)
    assert np.all(vec_combine(v0,v1) == mat_combine(v0,v1))
    