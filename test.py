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
            
    #%%
    grid = solver.model.grid
    
    # Plot to check
        
    # Plot contours
    fig, axs = plt.subplots(1, 3, figsize=(32, 13))

    cont1 = axs[0].imshow(grid.uField)
    plt.colorbar(cont1, location='bottom')
    axs[0].set_xlabel("X", fontsize=25)
    axs[0].set_ylabel("Y", fontsize=25)
    axs[0].set_title("u", fontsize=25)
        
    cont2 = axs[1].imshow(grid.vField)
    plt.colorbar(cont2, location='bottom')
    axs[1].set_xlabel("X", fontsize=25)
    axs[1].set_title("v", fontsize=25)

    cont3 = axs[2].imshow(grid.hField)
    plt.colorbar(cont3, location='bottom')
    # axs[2].contour(XS, YS, uSol, colors='black')
    axs[2].set_xlabel("X", fontsize=25)
    axs[2].set_title("$\eta$", fontsize=25)

    plt.show()