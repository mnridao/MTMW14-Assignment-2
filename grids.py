"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
    
class ArakawaCGrid:
    """
    """
    
    def __init__(self, xbounds, nx, ybounds=None, ny=None, periodicX=False):
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
        
        # Check for periodic boundary condition.
        self.periodicX = periodicX
        self.periodicY = False       # Remove this option for now.
        
        # Set up arrays representing the 2D grid.
        self.createGrid()
                
        # Initialise the default state variables.
        self.resetFields()
    
    def copy(self):
        """ 
        """
        
        # Create grid object.
        grid = ArakawaCGrid(self.xbounds, self.nx, self.ybounds, self.ny, 
                            self.periodicX)
        
        # Copy fields.
        grid.uField = self.uField.copy()
        grid.vField = self.vField.copy()
        grid.hField = self.hField.copy()
        
        # Store by ref in dictionary for iterating over fields.
        grid.fields = {"uVelocity" : grid.uField if grid.periodicX else grid.uField[:, 1:-1],
                       "vVelocity" : grid.vField if grid.periodicY else grid.vField[1:-1, :],
                       "eta"       : grid.hField}
        
        return grid
            
    def createGrid(self):
        """ 
        """
        
        # Number of points in grid (depends on boundary conditions).
        numX = self.nx + 1*(not self.periodicX)
        numY = self.ny + 1*(not self.periodicY)
        
        # Create full-point grid (exclude last point if periodic).
        xpoints = np.linspace(self.xbounds[0], self.xbounds[1], numX, 
                              not self.periodicX)
        ypoints = np.linspace(self.ybounds[0], self.ybounds[1], numY, 
                              not self.periodicY)
        
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
        
        # Number of points in grid (depends on boundary conditions).
        numX = self.nx + 1*(not self.periodicX)
        numY = self.ny + 1*(not self.periodicY)
        
        # Initialise default state variables.
        self.uField = np.zeros(shape=(self.ny, numX))
        self.vField = np.zeros(shape=(numY, self.nx))
        self.hField = np.zeros(shape=(self.nx, self.ny))
                
        # Store by ref in dictionary for iterating over fields.
        self.fields = {"uVelocity" : self.uField if self.periodicX else self.uField[:, 1:-1],
                       "vVelocity" : self.vField if self.periodicY else self.vField[1:-1, :],
                       "eta"       : self.hField}
        
        # Extra field for bottom topography.
        self.hBot = np.zeros_like(self.hField)
    
    def uGrid(self):
        """ 
        """
        return (self.X[:-1, :], np.hstack((self.Ymid, self.Ymid[:, 0].reshape(-1, 1))))
        
    def vGrid(self):
        """ 
        """
        return (np.vstack((self.Xmid, self.Xmid[0, :].reshape(1, -1))), self.Y[:, :-1])
        
    def etaGrid(self):
        """ 
        """
        return (self.Xmid, self.Ymid)
    
    def dudxField(self):
        """ 
        """
        
        return self.forwardGradientFieldX(self.uField)
            
    def dvdyField(self):
        """ 
        """
        
        return self.forwardGradientFieldY(self.vField)
        
    def detadxField(self):
        """ 
        """
        
        return self.backwardGradientFieldX(self.hField)
    
    def detadyField(self):
        """ 
        """
        return self.backwardGradientFieldY(self.hField)
    
    def vorticityField(self):
        """ 
        """
        return (self.forwardGradientFieldX(self.vField) - 
                self.forwardGradientFieldY(self.uField))
    
    def uOnEtaField(self):
        """ 
        """
        return 0.5*(self.uField[:, :-1] + self.uField[:, 1:])
        
    def vOnEtaField(self):
        """ 
        """
        return 0.5*(self.vField[:-1, :] + self.vField[1:, :])
    
    def vOnUField(self):
        """ 
        """
        
        # If using reflective boundary conditions we only want interior points.
        if not (self.periodicX or self.periodicY):
            return self.interpolateInterior(self.vField)
                
        # Check for periodic boundary conditions in X axis.
        if self.periodicX:
            
            # Make sure field is the same size as u-field.
            field = np.zeros_like(self.uField)
                        
            # Interpolate interior points.
            field[:, 1:] = self.interpolateInterior(self.vField)
            
            # Interpolate edge points (last point instead????).
            field[:, 0] = (self.vField[:-1, -1] + self.vField[1:, -1] + 
                           self.vField[:-1, 0]  + self.vField[1:, 0])/4
                
        return field
    
    def uOnVField(self):
        """ 
        """
        
        # If using reflective boundary conditions we only want interior points.
        if not (self.periodicY or self.periodicX):
            return self.interpolateInterior(self.uField)
                
        elif self.periodicX:
            
            # Interior interpolation will miss out the last col in v-field.
            field = np.zeros_like(self.fields["vVelocity"])
            
            # Interpolate interior points.
            field[:, :-1] = self.interpolateInterior(self.uField)
            
            # Interpolate last point.
            field[:, -1] = (self.uField[:-1, 0] + self.uField[:-1, -1] + 
                            self.uField[1:, 0]  + self.uField[1:, -1])/4
        
        return field
    
    def forwardGradientFieldX(self, field):
        """ 
        """
        
        # Calculates field based on periodic boundary conditions.
        if self.periodicX:
            return (np.roll(field, -1, 1) - field)/self.dx
        
        # Calculates field based on reflective boundary conditions.
        return (field[:, 1:] - field[:, :-1])/self.dx
    
    def backwardGradientFieldX(self, field):
        """ 
        """
        
        # Calculates field based on periodic boundary conditions.
        if self.periodicX:
            return (field - np.roll(field, 1, 1))/self.dx
        
        # Calculates field based on reflective boundary conditions. (same?)
        return (field[:, 1:] - field[:, :-1])/self.dx # i think ok
    
    def forwardGradientFieldY(self, field):
        """ 
        """
        
        # Calculates field based on periodic boundary conditions.
        if self.periodicY:
            return (np.roll(field, -1, 0) - field)/self.dy
        
        # Calculates field based on reflective boundary conditions.
        return (field[1:, :] - field[:-1, :])/self.dy
    
    def backwardGradientFieldY(self, field):
        """ 
        """
        # Calculates field based on periodic boundary conditions.
        if self.periodicY:
            return (field - np.roll(field, 1, 0))/self.dy
        
        # Calculates field based on reflective boundary conditions. (same?)
        return (field[1:, :] - field[:-1, :])/self.dy # i think ok
    
    def interpolateInterior(self, phi):
        """ 
        """
        return (phi[:-1, :-1] + phi[1:, :-1] + phi[:-1, 1:] + phi[1:, 1:])/4