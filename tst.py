#!/usr/bin/env python3

# this is for whatever

'''
from multiprocessing import Pool

def power(x, n=10):
    return x**n
 
    


pool = Pool()
pow10 = pool.map(power, range(10,20))
print(pow10)
'''

from __future__ import print_function
from scoop import futures

def helloWorld(value):
    return "Hello World from Future #{0}".format(value)

if __name__ == "__main__":
    returnValues = list(futures.map(helloWorld, range(16)))
    print("\n".join(returnValues))