"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

def calculateEnergyModel(model):
    """ 
    Calculates the energy 
    
    Inputs
    -------
    
    Returns
    -------
    """
    # Get the problem parameters from one of the equations in the model.
    params = model.eqns[0].params
    
    # Interpolate u and v onto eta grid.
    u = 

    return (np.sum(0.5*params.rho*(model.grid.uField[:, :-1]**2 + 
                                   model.grid.vField[:-1, :]**2 + 
                                   params.g*model.grid.hField**2))* 
            model.grid.dx**2)

def calculateEnergy(u, v, eta, dx, params):
    """
    Calculates the energy
    
    Inputs
    -------
    
    Returns
    -------
    """
    
    return (np.sum(0.5*params.rho*(u**2 + v**2 + params.g*eta**2))*dx**2)

def calculateTimestepCFL(c, d):
    """ 
    Calculates and returns the timestep based on the 2D CFL criterion. The 
    calculated timestep is rounded down to be extra safe.
    
    Inputs
    -------
    c : float 
        Speed used  to calculate CFL. In our application we are limited by the 
        speed of gravity waves, since these are the fastest waves we can expect 
        to encounter.
    d : float
        The grid spacing used to calculate CFL. In the case of a mesh with 
        variable spacing, this should be the smallest grid spacing.
    
    Returns
    -------
    timestep : float
        The calculated timestep based on the CFL criterion.
    """
    return np.floor(d/(c*np.sqrt(2)))