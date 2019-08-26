# These are interpolation routines I use
# sorted refers to strictly increasing
# feeding non-strictly-increasing data may result in unpredictable behavior
import numpy as np
from numba import jit
import pytest


no_py = True


@jit(nopython=no_py)
def interpolate_vv(x,xd,y,istart,testorder=False,xd_ordered=True):
    # this one was written by Matthew Rognlie at Northwestern
    # I modify it by including starting value
    # some syntax is a bit odd to use (nopython=True)
    
    if testorder:
        assert np.all(np.diff(x)>0), 'sample xs not ordered'
        if xd_ordered:
            assert np.all(np.diff(xd)>0), 'desired xs not ordered'
    
    
    xdi = np.empty(xd.size, np.int32)
    xdpi = np.empty(xd.size, np.float32)
    yd = np.empty(xd.size, np.float32)
    
    nxd,nx = (xd.shape[0],x.shape[0])
    
    assert istart.ndim==1
    
    assert istart.size == xd.size, 'wrong size of istart'
    
    xi = istart[0] # note that istart has to be nonempty
            
    xi_assign = 0
    
    for xdi_cur in range(nxd):
        xd_cur = xd[xdi_cur]
        
        if not xd_ordered:
            xi = 0 # reset
            
        # replace if have starting guess
        if istart[xdi_cur] > xi:
            xi = istart[xdi_cur]
            
            
        while xi < nx-1:
            
            if x[xi] >= xd_cur:
                break
            xi += 1
            
        if xi != 0:
            xi_assign = np.int32(xi-1)
        else:
            xi_assign = np.int32(0)
            
        xdpi_cur = (x[xi_assign+1]-xd_cur) / (x[xi_assign+1]-x[xi_assign])
        
        yd[xdi_cur] = xdpi_cur*y[xi_assign] + (1-xdpi_cur)*y[xi_assign+1]
        
        xdpi[xdi_cur] = xdpi_cur
        xdi[xdi_cur] = xi_assign

    return xdi, xdpi, yd


def test_interpolate_vv():
    x = np.array([0,6,12],np.float32)
    y = np.array([5,4,3],np.float32)
    
    xq = np.array([3],np.float32)
    istart = np.array([0],np.int32)
    
    assert interpolate_vv(x,xq,y,istart)[2] == 4.5, 'one point is not ok'
    
    xq = np.array([-6],np.float32)
    assert interpolate_vv(x,xq,y,istart)[2] == 6, 'extrapolation is not ok'
    
    istart = np.array([0,0],np.int32)
    
    xq = np.array([9,15],np.float32)
    assert np.all(interpolate_vv(x,xq,y,istart)[2] == np.array([3.5,2.5])), 'vector is not ok'
    

@jit(nopython=no_py)
def interpolate_precomputed(xdi,xdpi,y):
    
    Yq = np.empty(xdi.shape,np.float32)
    
    for irow in range(0,Yq.shape[0]):
            xi  =  xdi[irow]
            xpi = xdpi[irow]
            Yq[irow] = xpi*y[xi] + (1-xpi)*y[xi+1]

    return Yq
# note: xdi is index in x, xdpi is the share of the next point, yd is interpolated value
  
@jit(nopython=no_py)
def interpolate_nostart(x,xd,y,testorder=False,xd_ordered=True):
    # NB: order is not checked
    istart = np.zeros(xd.shape, np.int32)
    
    xdi, xdpi, yd = interpolate_vv(x,xd,y,istart,testorder=testorder,xd_ordered=xd_ordered)
    
    return xdi, xdpi, yd
  
def interpolate(x,y,xq,axis=0,rsorted=True,csroted=True):
    # this is the main function
    # y, x can be: 
    # 1. vector, vector, then the same as interpolate_vv
    # 2. Nd array, vector. Then interpolation happens along the axis.
    # 3. vector, matrix, then interpolates y in each point of matrix
    # 4. matrix, matrix. then interpolates each column of y in each column of x
    # 5. list + whatever. then returns the list in which each element is 
    # interpolated as in case 1, 2, 3 or 4. 
    
    # highly experimental
    
    return 1

# this needs to be njited as array slicing is not supported somehow
@jit(nopython=no_py)
def interpolate_vector_matrix(x,y,xq,rsorted=True,csorted=True):
    # this interpolates vector y with domain given by x at each point of xq.
    # xq is assumed to be a matrix with at least rows sorted
    # this implementation is marginaly faster than scipy
    
    assert(rsorted and csorted)
    
    nrows = xq.shape[0]
    
    yq = np.empty(xq.shape)
    xdi = np.empty(xq.shape,np.int32)
    xdpi = np.empty(xq.shape)
    
    xdi[0,:],xdpi[0,:],yq[0,:] = interpolate_vv(x,xq[0,:],y,np.zeros(xq[0,:].shape,np.int32))
    
    for irow in range(1,nrows):
        xdi[irow,:],xdpi[irow,:],yq[irow,:] = interpolate_vv( x,xq[irow,:], y, xdi[irow-1,:])
    
    return yq




@jit#(nopython=True) # WHY WHY WHY
def interpolate_matrix_vector(x,Y,xq,startatzero=False):
    # this interpolates each column of Y with domain given by x at each point 
    # given by vector xq
    # 
    
    Yq = np.empty((xq.shape[0],Y.shape[1]),np.float32)
    
    
    
    xdi, xdpi, Yq[:,0] = interpolate_nostart(x,xq,Y[:,0])
    
    for j in range(1,Y.shape[1]):
        Yq[:,j] = interpolate_precomputed(xdi,xdpi,Y[:,j])
        
    return Yq


if __name__ == "__main__":
    pytest.main()

#g1 = np.array([1,4])
#g2 = np.array([[4,8],[12,24]])
#g3 = np.array([1.5,3.5])
#mv = interpolate_matrix_vector(g1,g2,g3)
#print(mv)