"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

# Functions used for the calculation of the analytical solutions.
f1 = lambda x, a, b: np.pi*(1 + ((np.exp(a) - 1)*np.exp(b*x) + 
                     (1 - np.exp(b))*np.exp(a*x))/(np.exp(b) - np.exp(a)))

f2 = lambda x, a, b: ((np.exp(a) - 1)*b*np.exp(b*x) + 
                     (1 - np.exp(b))*a*np.exp(a*x))/(np.exp(b) - np.exp(a))

def calculateEpsilon(params, L):
    """ 
    """
    return params.gamma / (L * params.beta)
    
def calculateA(params, L):
    """ 
    """
    epsilon = calculateEpsilon(params, L)
    return (-1 - np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    
def calculateB(params, L):
    """ 
    """
    epsilon = calculateEpsilon(params, L)
    return (-1 + np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)

def analyticalU(X, Y, L, params):
    """ 
    """
    a = calculateA(params, L)
    b = calculateB(params, L)
    
    return (- params.tau0 / (np.pi*params.gamma*params.rho*params.H) * 
            f1(X/L, a, b) * np.cos(np.pi*Y/L))
    
def analyticalV(X, Y, L, params):
    """ 
    """
    a = calculateA(params, L)
    b = calculateB(params, L)
    
    return (params.tau0/(np.pi*params.gamma*params.rho*params.H) * 
            f2(X/L, a, b) * np.sin(np.pi*Y/L))
    
def analyticalEta(X, Y, L, params, eta0=0.):
    """ 
    """
    a = calculateA(params, L)
    b = calculateB(params, L)
    
    return eta0 + (params.tau0/(np.pi*params.gamma*params.rho*params.H)* 
                   params.f0*L/params.g*(params.gamma/(params.f0*np.pi)*
                                         f2(X/L, a, b)*np.cos(np.pi*Y/L) + 
                                         f1(X/L, a, b)/np.pi*(np.sin(np.pi*Y/L)*
                                        (1 + params.beta*Y/params.f0) + 
                                        params.beta*L/(params.f0*np.pi)*
                                        np.cos(np.pi*Y/L))))

def analyticalSolution(grid, params, eta0=0.):
    """ 
    Returns the steady state analytical solution for an ocean gyre from 
    Mushgrave (1985). Calculates each variable on the respective grid of the 
    numerical solution, obtained from grid.
    
    Inputs
    -------
    grid   : ArakawaCGrid object
        Object containing the domain information of the problem. This is
        passed in so that the analytical solution can be calculated on the 
        same grid as the numerical solutions.
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
    
    L = grid.xbounds[1]
        
    # u-velocity grid (extra column added to Ymid).
    uSol = analyticalU(*grid.uGrid(), L, params)
    
    # v-velocity grid (extra row added to Xmid).
    vSol = analyticalV(*grid.vGrid(), L, params)
    
    # Eta grid.
    etaSol = analyticalEta(*grid.etaGrid(), L, params, eta0)
    
    return uSol, vSol, etaSol

# def analyticalSolution(X, Y, L, params, eta0=0.):
#     """ 
#     Returns the steady state analytical solution for an ocean gyre from 
#     Mushgrave (1985). 
    
    # Inputs
    # -------
    # X      : np array
    #     2D grid with x-domain coordinates.
    # Y      : np array
    #     2D grid with y-domain coordinates.
    # L      : float
    #     Upper bound of x and y domain (assumes both are equal).
    # params : Parameters object
    #     Object containing the default parameters of the problem.
    # eta0   : float or numpy array
    #     Unknown constant of integration. Default is 0.
    
    # Returns
    # -------
    # u   : np array
    #       Steady state analytical solution for u-velocity.
    # v   : np array
    #       Steady state analytical solution for v-velocity.
    # eta : np array
    #       Steady state analytical solution for height perturbation (eta).
#     """ 
    
#     # Functions used for the calculation of the analytical solutions.
#     f1 = lambda x, a, b: np.pi*(1 + ((np.exp(a) - 1)*np.exp(b*x) + 
#                          (1 - np.exp(b))*np.exp(a*x))/(np.exp(b) - np.exp(a)))
    
#     f2 = lambda x, a, b: ((np.exp(a) - 1)*b*np.exp(b*x) + 
#                          (1 - np.exp(b))*a*np.exp(a*x))/(np.exp(b) - np.exp(a))
    
#     # Terms used in the calculation of the analytical solutions.
#     epsilon = params.gamma / (L * params.beta)
#     a = (-1 - np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
#     b = (-1 + np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    
#     # Calculate u-velocity.
#     u = (- params.tau0 / (np.pi*params.gamma*params.rho*params.H) * 
#          f1(X/L, a, b) * np.cos(np.pi*Y/L))
    
#     # Calculate v-velocity.
#     v = (params.tau0/(np.pi*params.gamma*params.rho*params.H) * 
#          f2(X/L, a, b) * np.sin(np.pi*Y/L))
    
#     # Calculate eta (this is ugly no matter what, this is clear to me at least).
#     eta = eta0 + (params.tau0/(np.pi*params.gamma*params.rho*params.H)* 
#                   params.f0*L/params.g*(params.gamma/(params.f0*np.pi)*
#                                         f2(X/L, a, b)*np.cos(np.pi*Y/L) + 
#                                         f1(X/L, a, b)/np.pi*(np.sin(np.pi*Y/L)*
#                                         (1 + params.beta*Y/params.f0) + 
#                                         params.beta*L/(params.f0*np.pi)*
#                                         np.cos(np.pi*Y/L))
#                                         )
#                   )
    
#     return u, v, eta   # Don't really like this.