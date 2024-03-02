"""
MTMW14 Assignment 2

Student ID: 31827379
"""
import numpy as np
from scipy.stats import multivariate_normal

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
                    
    def setf0(self, f0):
        """ 
        """
        # Update f0 for each of the equations in the model.
        for eqn in self.eqns:
            eqn.params.f0 = f0
    
    def setBeta(self, beta):
        """ 
        """
        # Update beta for each of the equations in the model.
        for eqn in self.eqns:
            eqn.params.beta = beta
    
    def activateBetaPlane(self, activate):
        """ 
        """
        # Activate or deactivate the beta plane for all the model equations.
        for eqn in self.eqns:
            eqn.params.betaPlaneActive = activate
    
    def activateWindStress(self, activate):
        """
        """
        # Activate or deactivate the wind stress for all the model equations.
        for eqn in self.eqns:
            eqn.params.setWindStressX("default" if activate else "off")
            eqn.params.setWindStressY("default" if activate else "off")
    
    def setInitialCondition(self, key, *args):
        """ 
        """
        
        if key == "step":
                        
            self.setStepInitialCondition(*args)
        
        elif key == "blob":
            
            self.setBlobInitialCondition(*args)
        
        elif key == "reset":
            
            # Reset height perturbations to zero.
            self.grid.setDefaultEtaField()
            
    def setStepInitialCondition(self, X, Y, height):
        """ 
        """
        # Convert to indices.
        nx = (X/self.grid.dx).astype(int)
        ny = (Y/self.grid.dy).astype(int)
        
        # Update the appropriate fields.
        self.grid.hField[nx[0]:nx[1], ny[0]:ny[1]] = height
        
        # Update hField view - this is stupid.
        self.grid.fields["eta"] = self.grid.hField
        
    def setBlobInitialCondition(self, mu, var, height):
        """ 
        """
        # Create the Gaussian blob.
        pos = np.empty(self.grid.X.shape + (2,))
        pos[..., 0] = self.grid.X
        pos[..., 1] = self.grid.Y
        rv = multivariate_normal(mu, [[var[0], 0], [0, var[1]]])
        
        # Generate the blob height perturbation field.
        self.grid.hField = height * rv.pdf(pos)[:-1, :-1]
        
        # Update hField view - this is stupid.
        self.grid.fields["eta"] = self.grid.hField
    
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
        
        # Functions that should be called each iteration stored here. #
        # TODO: better way of doing this (accessing is difficult).  
        self.customEquations = []
        self.customData = []
        
    def run(self, *phi0):
        """ 
        """
        # Run the simulation for each time step.
        for t in range(self.nt):
            
            # Update the grid (no return since it was passed by reference).
            self.scheme(self.model.eqns, self.model.grid, self.dt, t)
            
            # Store state if necessary.
            # ...
            
            # Evaluate any functions added by user (e.g. energy)
            if self.customEquations:
                
                for i, customEqn in enumerate(self.customEquations):
                    
                    # Append the results of custom eqn to data list.
                    self.customData[i][t+1] = customEqn(self.model)
            
            # plotContourSubplot(self.model.grid)
            
    
    def runEnsemble(self, numEnsembles, perturbationRange, *phi0):
        """ 
        """
        pass
    
    def addCustomEquations(self, customEqn, numRes):
        """ 
        """
        # Initialise the custom equation and data field. 
        # TODO: better way of doing this.
        self.customEquations.append(customEqn)
        self.customData.append(np.zeros(shape=(self.nt+1, numRes)))
        
        # Add initial return result.
        self.customData[-1][0] = self.customEquations[-1](self.model)