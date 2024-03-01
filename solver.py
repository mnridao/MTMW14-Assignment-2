"""
MTMW14 Assignment 2

Student ID: 31827379
"""
from plotters import plotContourSubplot

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
        
        # TODO: attach functions here e.g. energy, to be calculated each iteration?
        # frames for plots as well?
        self.monitors = []
        
class Solver:
    
    def __init__(self, model, scheme, dt, nt, store=False):
        """ 
        """        
        self.model   = model
        self.scheme  = scheme
        self.dt      = dt
        self.nt      = nt
        
        self.store   = store
        self.history = None
        self.ensembleHistory = None
                
    def run(self, *phi0):
        """ 
        """
        
        # Run the simulation for each time step.
        for t in range(self.nt):
            
            # Update the grid (no return since it was passed by reference).
            self.scheme(self.model.eqns, self.model.grid, self.dt, t)
            
            # Store state if necessary.
            # ...
            
            # Plt.
            plotContourSubplot(self.model.grid)
    
    def runEnsemble(self, numEnsembles, perturbationRange, *phi0):
        """ 
        """
        pass