"""
MTMW14 Assignment 2

Student ID: 31827379
"""
    
import numpy as np

def calculateEnergyModel(model):
    """ 
    Calculates the energy of the numerical solution. Energy is calculated at 
    the centroid of each cell (the eta grid), so velocities are interpolated.
    
    Inputs
    -------
    model : Model object 
            Object containing the problem equations and grid.
    
    Returns
    -------
    energy : float
       The energy calculated for the current state of the model.
    """
    
    # Interpolate u and v onto eta grid.
    u = 0.5*(model.grid.uField[:, :-1] + model.grid.uField[:, 1:])
    v = 0.5*(model.grid.vField[:-1, :] + model.grid.vField[1:, :])
    
    return calculateEnergy(u, v, model.grid.hField, 
                           model.grid.dx, model.eqns[0].params)

    # return (np.sum(0.5*params.rho*(model.grid.uField[:, :-1]**2 + 
    #                                model.grid.vField[:-1, :]**2 + 
    #                                params.g*model.grid.hField**2))* 
    #         model.grid.dx**2)

def calculateEnergy(u, v, eta, dx, params):
    """
    Calculates the energy of either the numerical or analytical solution.
    
    Inputs
    -------
    u      : np array
             Array containing the u-velocity values.
    v      : np array
             Array containing the v-velocity values.
    eta    : np array
             Array containing the eta values.
    dx     : float
             Grid spacing.
    params : Parameters object.
             Object containing the parameter values for the problem.
    
    Returns
    -------
    energy : float
       The energy calculated for the current state of the model.
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