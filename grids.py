"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
from scipy.stats import multivariate_normal
    
class ArakawaCGrid:
    """
    """
    
    def __init__(self, xbounds, nx, ybounds=None, ny=None):
        """ 
        """
        
        # Number of grid points.
        self.nx = nx
        self.ny = ny if ny else nx
        
        # Domain bounds for 2D grid.
        self.xbounds = xbounds
        self.ybounds = ybounds if ybounds else xbounds
        
        # Calculate grid spacing from inputs.
        self.dx = (self.xbounds[1] - self.xbounds[0]) / self.nx
        self.dy = (self.ybounds[1] - self.ybounds[0]) / self.ny
        
        # Set up arrays representing the 2D grid.
        self.createGrid()
        
        # Initialise default state variables.
        self.uField = np.zeros(shape=(self.ny, self.nx+1))
        self.vField = np.zeros(shape=(self.ny+1, self.nx))
        self.hField = np.zeros(shape=(self.nx, self.ny))
        
        # Store by ref in dictionary for iterating over fields.
        self.fields = {"uVelocity" : self.uField[:, 1:-1],
                       "vVelocity" : self.vField[1:-1, :],
                       "eta"       : self.hField}
        
        # Set up beta plane.
        
        
    def createGrid(self):
        """ 
        """
        xpoints = np.linspace(self.xbounds[0], self.xbounds[1], self.nx+1)
        ypoints = np.linspace(self.ybounds[0], self.ybounds[1], self.ny+1)
        
        self.X, self.Y = np.meshgrid(xpoints, ypoints)
        
    def vOnUField(self):
        """ 
        """
        return self.interpolate(self.vField)
    
    def uOnVField(self):
        """ 
        """
        return self.interpolate(self.uField)
    
    def interpolate(self, phi):
        """ 
        """
        return (phi[:-1, :-1] + phi[1:, :-1] + phi[:-1, 1:] + phi[1:, 1:])/4
    
    def setInitialCondition(self, key, *args):
        """ 
        """
        
        if key == "step":
                        
            self.setStepInitialCondition(*args)
        
        elif key == "blob":
            
            self.setBlobInitialCondition(*args)
            
        # TODO: Should be able to reset the initial condition as well.
            
    def setStepInitialCondition(self, X, Y, height):
        """ 
        """
        # Convert to indices.
        nx = (X/self.dx).astype(int)
        ny = (Y/self.dy).astype(int)
        
        # Update the appropriate fields.
        self.hField[nx[0]:nx[1], ny[0]:ny[1]] = height
        
        # Update hField view - this is stupid.
        self.fields["eta"] = self.hField
        
    def setBlobInitialCondition(self, mu, var, height):
        """ 
        """
        # Create the Gaussian blob.
        pos = np.empty(self.X.shape + (2,))
        pos[..., 0] = self.X
        pos[..., 1] = self.Y
        rv = multivariate_normal(mu, [[var[0], 0], [0, var[1]]])
        
        # Generate the blob height perturbation field.
        self.hField = height * rv.pdf(pos)[:-1, :-1]
        
        # Update hField view - this is stupid.
        self.fields["eta"] = self.hField