"""
MTMW14 Assignment 2

Student ID: 31827379
"""
    
import numpy as np
import timeSchemes as schemes

from analyticalSolution import analyticalSolution

def setScheme(s, model=None, dt=None):
    """ 
    Set the scheme for the problem.
    
    Inputs
    -------
    s     : string
            Key for the required time scheme.
    model : Model object
            Object containing the model information (grid and equations).
    dt    : float
            Timestep.
    """
    if s == 1 or s == "forwardBackward":
        scheme = schemes.forwardBackwardSchemeCoupled
    elif s == 2 or s == "rk4":
        scheme = schemes.RK4SchemeCoupled
    elif s == 3 or s == "semiImplicit":
        scheme = schemes.SemiImplicitSchemeCoupled(model, dt)
    elif s == 4 or s == "semiLagrangian":
        scheme = schemes.SemiLagrangianSchemeCoupled()
    else: 
        return None
    return scheme

def calculateEnergyDifference(model):
    """ 
    Calculates the differences between the numerical model solution and the 
    analytical solution (assuming steady state has been reached) and computes
    the energy from the difference fields.
    
    Inputs
    -------
    model : Model object 
            Object containing the problem equations and grid.
            
    Returns
    -------
    energyDiff : float
       The energy calculated for the current state of the model.
    """
    
    # Calculate the analytical solution.
    eta0 = calculateEta0(model.grid.hField)
    uSS, vSS, etaSS = analyticalSolution(model.grid, model.eqns[0].params, eta0)
    
    # Calculate the difference fields between the model and analytical solution.
    uDiff = model.grid.uField - uSS
    vDiff = model.grid.vField - vSS
    hDiff = model.grid.hField - etaSS
    
    # Calculate the energy of the difference field.
    return calculateEnergy(uDiff, vDiff, hDiff, model.grid.dx, model.eqns[0].params)
    
def calculateEnergyModel(model):
    """ 
    Calculates the energy of the numerical solution. Assumes that the X and Y 
    domains are the same length, i.e. dx == dy.
    
    Inputs
    -------
    model : Model object 
            Object containing the problem equations and grid.
    
    Returns
    -------
    energy : float
       The energy calculated for the current state of the model.
    """
        
    return calculateEnergy(model.grid.uField, model.grid.vField, 
                           model.grid.hField, model.grid.dx, model.eqns[0].params)
    
def calculateEnergy(u, v, eta, dx, params):
    """
    Calculates the energy of either the numerical or analytical solution.
    Assumes that the X and Y domains are the same length, i.e. dx == dy.
    
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
    
    return 0.5*params.rho*dx**2*(params.H*(np.sum(u**2) + np.sum(v**2)) + 
                                 params.g*np.sum(eta**2))

def calculateEta0(hField):
    """ 
    Calculates the value of eta0 that should be used by the analytical 
    solution. Defined as eta0 = hField[-1, L/2] where L is the domain length, 
    and assuming that hField is the steady state distribution.
    
    Inputs
    ------
    hField : np array
             Array of eta values from the numerical solution.
    """
    
    mid = int(0.5*hField.shape[0])
    if hField.shape[0] % 2 == 0: 
        return 0.5*(hField[-1, mid-1] + hField[-1, mid])
    else:
        return hField[-1, mid]

def calculateTimestepCFL(c, d):
    """ 
    Calculates and returns the timestep based on the 2D CFL criterion. The 
    calculated timestep is rounded down to be extra safe. In our application we 
    are limited by the speed of gravity waves, since these are the fastest 
    waves we can expect to encounter.
    
    Inputs
    -------
    c : float 
        Speed used  to calculate CFL.
    d : float
        The grid spacing used to calculate CFL. In the case of a mesh with 
        variable spacing, this should be the smallest grid spacing.
    
    Returns
    -------
    timestep : float
        The calculated timestep based on the CFL criterion.
    """
    return np.floor(d/(c*np.sqrt(2)))

def runAllSchemesForNDays(solver, N):
    """ 
    """
    
    # Set up the new number of time steps for the solver.
    endtime = N*24*60**2
    solver.nt = int(np.ceil(endtime/solver.dt))
    
    # Define the schemes that will be run.
    s = ["forwardBackward", "rk4", "semiLagrangian", "semiImplicit"]
    
    # Initialise storage arrays.
    hFields = np.zeros(shape=(len(s), *solver.model.grid.hField.shape))
    uFields = np.zeros(shape=(len(s), *solver.model.grid.uField.shape))
    vFields = np.zeros(shape=(len(s), *solver.model.grid.vField.shape))
    
    for i, si in enumerate(s):
        # Reset the grid fields.
        solver.model.grid.resetFields()
        
        # Set the current scheme and run (extra args for si).
        solver.scheme = setScheme(si, solver.model, solver.dt)
        solver.run()
        
        # Save the field information.
        hFields[i, ...] = solver.model.grid.hField 
        uFields[i, ...] = solver.model.grid.uField
        vFields[i, ...] = solver.model.grid.vField
        
    return hFields, uFields, vFields