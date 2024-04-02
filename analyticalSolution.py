"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

def analyticalSolution(X, Y, L, params, eta0=0.):
    """ 
    Returns the steady state analytical solution for an ocean gyre from 
    Mushgrave (1985). 
    
    Inputs
    -------
    X      : np array
        2D grid with x-domain coordinates.
    Y      : np array
        2D grid with y-domain coordinates.
    L      : float
        Upper bound of x and y domain (assumes both are equal).
    params : Parameters object
        Object containing the default parameters of the problem.
    eta0   : float or numpy array
        Unknown constant of integration. Default is 0.
    
    Returns
    -------
    u   : np array
          Steady state analytical solution for u-velocity.
    v   : np array
          Steady state analytical solution for v-velocity.
    eta : np array
          Steady state analytical solution for height perturbation (eta).
    """ 
    
    # Functions used for the calculation of the analytical solutions.
    f1 = lambda x, a, b: np.pi*(1 + ((np.exp(a) - 1)*np.exp(b*x) + 
                         (1 - np.exp(b))*np.exp(a*x))/(np.exp(b) - np.exp(a)))
    
    f2 = lambda x, a, b: ((np.exp(a) - 1)*b*np.exp(b*x) + 
                         (1 - np.exp(b))*a*np.exp(a*x))/(np.exp(b) - np.exp(a))
    
    # Terms used in the calculation of the analytical solutions.
    epsilon = params.gamma / (L * params.beta)
    a = (-1 - np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    b = (-1 + np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    
    # Calculate u-velocity.
    u = (- params.tau0 / (np.pi*params.gamma*params.rho*params.H) * 
         f1(X/L, a, b) * np.cos(np.pi*Y/L))
    
    # Calculate v-velocity.
    v = (params.tau0/(np.pi*params.gamma*params.rho*params.H) * 
         f2(X/L, a, b) * np.sin(np.pi*Y/L))
    
    # Calculate eta (this is ugly no matter what, this is clear to me at least).
    eta = eta0 + (params.tau0/(np.pi*params.gamma*params.rho*params.H)* 
                  params.f0*L/params.g*(params.gamma/(params.f0*np.pi)*
                                        f2(X/L, a, b)*np.cos(np.pi*Y/L) + 
                                        f1(X/L, a, b)/np.pi*(np.sin(np.pi*Y/L)*
                                        (1 + params.beta*Y/params.f0) + 
                                        params.beta*L/(params.f0*np.pi)*
                                        np.cos(np.pi*Y/L))
                                        )
                  )
    
    return u, v, eta   # Don't really like this.