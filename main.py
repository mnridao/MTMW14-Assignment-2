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

# # Energy function for TASK E
# def calculateEnergy(u, v, eta):
#     return np.sum(0.5*rho*(u[:, :-1]**2 + v[:-1, :]**2 + g*eta**2))*d**2

if __name__ == "__main__":
        
    # Grid creation.
    x0, xL = 0, 1e6 
    dx = 12.5e3
    nx = int((xL - x0)/dx)
    grid = ArakawaCGrid([x0, xL], nx)
    grid.setInitialCondition("blob", [5e5, 5e5], [dx**2, dx**2], 1000*dx)
    plotContourSubplot(grid)
    
    #%%
    # Time stepping information.
    dt = 175/2
    endtime = 10*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    # Set up the model and solver.
    scheme = forwardBackwardSchemeCoupled
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    solver = Solver(model, scheme, dt, nt)
    
    #%% Task D
    solver.run()
    plotContourSubplot(solver.model.grid)