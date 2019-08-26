# -*- coding: utf-8 -*-
'''
This code is borrowed from Jeppe Druedahl on Aug 2, 2019
Full repo link is: https://github.com/NumEconCopenhagen/ConsumptionSaving
'''

"""upperenvelope
This module provides a Numba JIT compilled upper envelope for a custom utility function.
"""

import numpy as np
from numba import njit

def create(ufunc,use_inv_w=False):
    """ create upperenvelope function from the utility function ufunc
    
    Args:
        ufunc (callable): utility function with *args (must be decorated with @njit)
    Returns:
        upperenvelope (callable): upperenvelope called as (grid_a,m_vec,c_vec,inv_w_vec,use_inv_w,grid_m,c_ast_vec,v_ast_vec,*args)
        use_inv_w (bool,optional): assume that the post decision value-of-choice vector is a negative inverse
    
    """
    
    #ufunc = njit(ufunc)
    
    @njit
    def upperenvelope(grid_a,m_vec,c_vec,inv_w_vec,grid_m,c_ast_vec,v_ast_vec,*args):
        """ upperenvelope function
        
        Args:
            grid_a (numpy.ndarray): input, end-of-period asset vector of length Na
            m_vec (numpy.ndarray): input, cash-on-hand vector from egm of length Na
            c_vec (numpy.ndarray): input, consumption vector from egm of length Na
            inv_w_vec (numpy.ndarray): input, post decision value-of-choice vector from egm of length Na
            grid_m (numpy.ndarray): input, common grid for cash-on-hand of length Nm
            c_ast_vec (numpy.ndarray): output, consumption on common grid for cash-on-hand of length Nm
            v_ast_vec (numpy.ndarray): output, value-of-choice on common grid for cash-on-hand of length Nm
            *args: additional arguments to the utility function
                    
        """

        # for given m_vec, c_vec and w_vec (coming from grid_a)
        # find the optimal consumption choices (c_ast_vec) at the common grid (grid_m) 
        # using the upper envelope + also value the implied values-of-choice (v_ast_vec)

        Na = grid_a.size
        Nm = grid_m.size

        c_ast_vec[:] = 0
        v_ast_vec[:] = -np.inf

        # constraint
        # the constraint is binding if the common m is smaller
        # than the smallest m implied by EGM step (m_vec[0])

        im = 0
        while im < Nm and grid_m[im] <= m_vec[0]:
            
            # a. consume all
            c_ast_vec[im] = grid_m[im] 

            # b. value of choice
            u = ufunc(c_ast_vec[im],*args)
            if use_inv_w:
                v_ast_vec[im] = u + (-1.0/inv_w_vec[0])
            else:
                v_ast_vec[im] = u + inv_w_vec[0]

            im += 1
            
            
        # fix at maximum m
        
        

        # upper envellope
        # apply the upper envelope algorithm
        
        for ia in range(Na-1):

            # a. a inteval and w slope
            a_low  = grid_a[ia]
            a_high = grid_a[ia+1]
            
            inv_w_low  = inv_w_vec[ia]
            inv_w_high = inv_w_vec[ia+1]

            
            inv_w_slope = (inv_w_high-inv_w_low)/(a_high-a_low)
            
            # b. m inteval and c slope
            m_low  = m_vec[ia]
            m_high = m_vec[ia+1]


            #if m_low > m_high:
            #    continue
            
            
            c_low  = c_vec[ia]
            c_high = c_vec[ia+1]

            c_slope = (c_high-c_low)/(m_high-m_low)

            # c. loop through common grid
            for im in range(Nm):

                # i. current m
                m = grid_m[im]

                # ii. interpolate?
                interp = (m >= m_low) and (m <= m_high)            
                extrap_above = ia == Na-2 and m > m_vec[Na-1]

                # iii. interpolation (or extrapolation)
                if interp or extrap_above:
                    
                    
                    # o. implied guess
                    c_guess = c_low + c_slope * (m - m_low)
                    
                    #assert c_guess > 0
                    
                    a_guess =  m - c_guess
                    #a_guess = np.maximum(  m - c_guess, grid_a[0] )
                    #c_guess = m - a_guess
                    
                    
                    #assert not(np.isnan(c_guess)) and (c_guess > 0)
                    
                    # oo. implied post-decision value function
                    inv_w = inv_w_low + inv_w_slope * (a_guess - a_low)  
                    
                    if a_guess < grid_a[0]:  inv_w = -np.inf 
                    if a_guess >= grid_a[-1]: inv_w = inv_w_vec[-1]

                    # ooo. value-of-choice
                    u = ufunc(c_guess,*args) if c_guess > 0 else -np.inf
                    
                    
                    
                    
                    if use_inv_w:
                        v_guess = u + (-1/inv_w)
                    else:
                        v_guess = u + inv_w

                    # oooo. update
                    if v_guess > v_ast_vec[im]:
                        v_ast_vec[im] = v_guess
                        c_ast_vec[im] = c_guess
                        assert m > 0
                        assert c_ast_vec[im] >= 0
                       
        
        #for im
        '''
        if c_ast_vec[-1] == 0:
            c_slope = (c_ast_vec[-2] - c_ast_vec[-3]) / ( grid_m[-2] - grid_m[-3] )
            v_slope = (v_ast_vec[-2] - v_ast_vec[-3]) / ( grid_a[-2] - grid_a[-3] )
            c_ast_vec[-1] = c_ast_vec[-2] + c_slope*( grid_m[-1] - grid_m[-2] )
            v_ast_vec[-1] = v_ast_vec[-2] + v_slope*( grid_a[-1] - grid_a[-2] )
        '''
    return upperenvelope