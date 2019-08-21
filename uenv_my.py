# This is implementation of upper envelope algoithm done by me

# variable size arrays are used. code is njitable
import numpy as np

def ue(m,c,v,u):
    
    j = 1
    while j < m.size:
        G = m.size
        if m[j] > m[j-1]:
            j += 1
            continue
        else:
            # elements by j-1 inclusive (slice :j) are normal
            for h in range(j,G-1):
                if m[h] < m[h+1]: break
            
            #if h is not found:
            if h == G-2 and m[h] > m[h+1]:
                m = m[:j]
                c = c[:j]
                v = v[:j]
                
            #if h is found what we care is on interval between j and h
            
            J_pre = slice(0,j,1)
            J_mid = slice(j-1,h+1,1)
            J_aft = slice( )
            
                
        
        
        
    
    
    