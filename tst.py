#!/usr/bin/env python3

# this is for whatever

from multiprocessing import Pool

def power(x, n=10):
    return x**n
 
pool = Pool()
pow10 = pool.map(power, range(10,20))
print(pow10)