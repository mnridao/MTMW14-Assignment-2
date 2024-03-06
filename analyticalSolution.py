"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

""" 
"""
f1 = lambda x, a, b: np.pi*(1 + ((np.exp(a) - 1)*np.exp(b*x) + 
                     (1 - np.exp(b))*np.exp(a*x))/(np.exp(b) - np.exp(a)))

f2 = lambda x, a, b: ((np.exp(a) - 1)*b*np.exp(b*x) + 
                     (1 - np.exp(b))*a*np.exp(a*x))/(np.exp(b) - np.exp(a))

def analyticalSolution(X, Y, L, params):
    """ 
    """ 
    
    epsilon = params.gamma / (L * params.beta)
    a = (-1 - np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    b = (-1 + np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    
    # Calculate u-velocity.
    u = (- params.tau0 / (np.pi*params.gamma*params.rho*params.H) * 
         f1(X/L, a, b) * np.cos(np.pi*Y/L))
    
    # Calculate v-velocity.
    v = (params.tau0/(np.pi*params.gamma*params.rho*params.H) * 
         f2(X/L, a, b) * np.sin(np.pi*Y/L))
    
    # Calculate eta.
    eta = (params.tau0/(np.pi*params.gamma*params.rho*params.H) * 
           params.f0*L/params.g * (params.gamma/(params.f0*np.pi) * 
           f2(X/L, a, b)*np.cos(np.pi*Y/L) + f1(X/L, a, b)/np.pi * 
           (np.sin(np.pi*Y/L)*(1 + params.beta*Y/params.f0) + params.beta*L /
           (params.f0*np.pi)*np.cos(np.pi*Y/L))))
    
    return u, v, eta   # Don't really like this.