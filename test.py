"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt

from equations import UVelocity, VVelocity, Eta
from grids import ArakawaCGrid
from timeSchemes import forwardBackwardSchemeCoupled

class Model:
    """ 
    """
    def __init__(self, eqns, grid):
        """ 
        """
        # Model equations and domain.
        self.eqns = eqns
        self.grid = grid
        
        # TODO: add functions to change this via this class.
        self.windStressActivated = True
        self.betaPlaneActivated  = True

class Solver:
    """ 
    """
    def __init__(self, model, scheme, dt, nt, store=False):
        """ 
        """        
        self.model   = model
        self.scheme  = scheme
        self.dt      = dt
        self.nt      = nt
        
        self.store   = store
        self.history = None
        
    def run(self):
        """ 
        """
        
        # Run the simulation for each time step.
        for t in range(self.nt):
            
            # Update the grid (no return since it was passed by reference).
            self.scheme(self.model.eqns, self.model.grid, self.dt, t)
            
            # Store state if necessary.
            # ...
            
    def runEnsemble(self):
        """ 
        """
        
        pass

if __name__ == "__main__":
    
    # Grid spacing information.
    x0 = 0 
    xL = 1e6 
    dx = 50e3
    nx = int((xL - x0)/dx)
    
    # Initialise grid
    grid = ArakawaCGrid([x0, xL], nx)
    
    # Time stepping information.
    dt = 350
    endtime = 10*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    scheme = forwardBackwardSchemeCoupled
        
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    solver = Solver(model, scheme, dt, nt)
    solver.run()
            