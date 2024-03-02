"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

from solver import Solver, Model
from equations import UVelocity, VVelocity, Eta
from grids import ArakawaCGrid
from timeSchemes import forwardBackwardSchemeCoupled
from plotters import plotContourSubplot

def createGrid(xbounds, dx, ybounds, d):
    """ 
    """
    pass

def setupModel():
    """ 
    """
    pass

def setupSolver():
    """ 
    """
    pass

def calculateEnergy(model):
    """ 
    """
    params = model.eqns[0].params
    return (np.sum(0.5*params.rho*(model.grid.uField[:, :-1]**2 + 
                                  model.grid.vField[:-1, :]**2 + 
                                  params.g*model.grid.hField**2)) * 
            model.grid.dx**2)

if __name__ == "__main__":
        
    # Grid creation.
    x0, xL = 0, 1e6
    dx = 50e3
    nx = int((xL - x0)/dx)
    grid = ArakawaCGrid([x0, xL], nx)
    
    #%%
    # Time stepping information.
    dt = 350
    endtime = 10*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    # Set up the model and solver.
    scheme = forwardBackwardSchemeCoupled
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    solver = Solver(model, scheme, dt, nt)
    
    # Add energy calculator to solver.
    solver.addCustomEquations(calculateEnergy, 1)
    
    #%%
    # solver.model.setInitialCondition("step", np.array([0.5, 0.55*xL]), np.array([0.*xL, 0.05*xL]), 100*dx)
    # plotContourSubplot(solver.model.grid)
    
    #%% Task D
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Turn rotation on/off.
    solver.model.activateBetaPlane(False)
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Turn wind on/off.
    solver.model.activateWindStress(True)
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Monitors.
    
    # solver.model