"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
    
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
                
        # Initialise the default state variables.
        self.resetFields()
        
    def createGrid(self):
        """ 
        """
        
        # Create full-point grid.
        xpoints = np.linspace(self.xbounds[0], self.xbounds[1], self.nx+1)
        ypoints = np.linspace(self.ybounds[0], self.ybounds[1], self.ny+1)
        
        self.X, self.Y = np.meshgrid(xpoints, ypoints)
        
        # Create mid-point grid.
        xpointsHalf = np.linspace(self.xbounds[0]+0.5*self.dx, 
                                  self.xbounds[1]-0.5*self.dx, self.nx)
        ypointsHalf = np.linspace(self.ybounds[0]+0.5*self.dy,
                                  self.ybounds[1]-0.5*self.dy, self.ny)
        
        self.Xmid, self.Ymid = np.meshgrid(xpointsHalf, ypointsHalf)

        
    def resetFields(self):
        """ 
        """
        # Initialise default state variables.
        self.uField = np.zeros(shape=(self.ny, self.nx+1))
        self.vField = np.zeros(shape=(self.ny+1, self.nx))
        self.hField = np.zeros(shape=(self.nx, self.ny))
        
        # Store by ref in dictionary for iterating over fields.
        self.fields = {"uVelocity" : self.uField[:, 1:-1],
                       "vVelocity" : self.vField[1:-1, :],
                       "eta"       : self.hField}
    
    def dudxField(self):
        """ 
        """
        pass
        
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