#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this approximates continuous deterministic transition with
# transition on grid


def transition_uniform_scalar(grid,xnew):
    import numpy as np

    # scalar version
    
    assert np.all(np.diff(grid)>0), "Grid must be strictly increasing"
    
    value_to = np.minimum(grid[-1], np.maximum(grid[0], xnew) )
    
    
    j_to = np.maximum( np.minimum( np.sum(value_to > grid)-1, grid.size - 2), 0)
    
    p_next = (value_to - grid[j_to]) / (grid[j_to+1] - grid[j_to])
    p_to = 1 - p_next
    
    assert (p_to<=1 and p_to>=0)    
    
    return j_to, p_to
    


def transition_uniform_looped(grid,xnew):
    import numpy as np

    j_to = np.empty(xnew.shape,np.int32)
    p_to = np.empty(xnew.shape,np.float32)
    
    for i in range(j_to.size):
        j_to[i], p_to[i] = transition_uniform_scalar(grid,xnew[i])
        
    return j_to, p_to
    

def transition_uniform(grid,xnew):
    import numpy as np

    
    assert grid.ndim==1, "grid should be 1-dimensional array"
    assert xnew.ndim==1, "xnew should be 1-dimensional array"
    
    j_to = np.empty(xnew.shape,np.int32)
    p_to = np.empty(xnew.shape,np.float32)
    
    value_to = np.minimum(grid[-1], np.maximum(grid[0], xnew) )
    
    
    
    j_to[:] = np.maximum( np.minimum(  np.sum(value_to[:,np.newaxis] > grid, axis=1) - 1, grid.size - 2 ), 0 )
    
    p_next = (value_to - grid[j_to]) / (grid[j_to+1] - grid[j_to])
    p_to[:] = 1 - p_next
    
    assert (np.all(p_to<=1) and np.all(p_to>=0))
    
    return j_to, p_to
    



if __name__ == "__main__":
    import numpy as np

    grid = np.array([1,2,3,5,6,9])
    xnew = np.array([2.5,2.5,1.2])

    j, p = transition_uniform(grid,xnew)

    u = transition_uniform(grid,xnew)
    f = transition_uniform_looped(grid,xnew)

    assert np.all(u[0] == f[0]) and np.all( np.abs(u[1]-f[1]) < 1e-6 ), "Different results?"

    print( (j,p) )