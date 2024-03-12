"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
    
class ArakawaCGrid:
    """
    """
    
    def __init__(self, xbounds, nx, ybounds=None, ny=None, 
                 periodicX=False, periodicY=False):
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
        
        # Store boundary condition.
        self.periodicX = periodicX
        self.periodicY = periodicY
        
        # Set up arrays representing the 2D grid.
        self.createGrid()
                
        # Initialise the default state variables.
        self.resetFields()
        
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
        
        # TODO: Update the views below for periodic boundaries.
        
        # Store by ref in dictionary for iterating over fields.
        self.fields = {"uVelocity" : self.uField[:, 1:-1],
                       "vVelocity" : self.vField[1:-1, :],
                       "eta"       : self.hField}
    
    def dudxField(self):
        """ 
        """
        
        return self.gradientFieldX(self.uField)
            
    def dvdyField(self):
        """ 
        """
        
        return self.gradientFieldY(self.vField)
        
    def detadxField(self):
        """ 
        """
        
        return self.gradientFieldX(self.hField)
    
    def detadyField(self):
        """ 
        """
        
        return self.gradientFieldY(self.hField)
    
    def vOnUField(self):
        """ 
        """
        
        # What happens if both periodic????????????
        if self.periodicX and self.periodicY:
            
            # Make sure field is the same size as u-field.
            field = np.zeros_like(self.uField)
            
            # Interpolate interior points.
            field[:-1, :-1] = self.interpolateInterior(self.vField)
            
            # Interpolate edge points.
            
        
        # Check for periodic boundary conditions in X axis.
        if self.periodicX and not self.periodicY:
            
            # Make sure field is the same size as u-field.
            field = np.zeros_like(self.uField)
            
            # Interpolate interior points.
            field[:, 1:] = self.interpolateInterior(self.vField)
            
            # Interpolate edge points (last point instead????).
            field[:, 0] = (self.vField[:-1, -1] + self.vField[1:, -1] + 
                           self.vField[:-1, 0]  + self.vField[1:, 0])/4
        
        elif self.periodicY:
            # Interior interpolation will miss out the last row in u-field.
            field = np.zeros_like(self.fields["uVelocity"])
            
            # Interpolate interior points.
            field[:-1, :] = self.interpolateInterior(self.vField)
            
            # Interpolate last point.
            field[-1, :] = (self.vField[0, 1:] + self.vField[0, :-1] + 
                            self.vField[-1, 1:]  + self.vField[-1, :-1])/4
        
        elif not (self.periodicX or self.periodicY):
            return self.interpolateInterior(self.vField)
    
    def uOnVField(self):
        """ 
        """
        
        # What happens if both periodic???
        if self.periodicX and self.periodicY:
            
            # Make sure field is the same size as v-field.
            field = np.zeros_like(self.vField)
            
            # Interpolate interior points.
            field[:-1, :-1] = self.interpolateInterior(self.uField)
            
            # Interpolate edge points.
            
        
        # Check for periodic boundary conditions in Y axis.
        elif self.periodicY and not self.periodicX:
            
            # Make sure field is the same size as v-field.
            field = np.zeros_like(self.vField)
            
            # Interpolate interior points.
            field[1:, :] = self.interpolateInterior(self.uField)
            
            # Interpolate edge points.
            field[0, :]  = (self.uField[0, :-1] + self.uField[0, 1:] + 
                            self.uField[-1, :-1] + self.uField[-1, 1:])/4
        
        elif self.periodicX:
            
            # Interior interpolation will miss out the last col in v-field.
            field = np.zeros_like(self.fields["vVelocity"])
            
            # Interpolate interior points.
            field[:, :-1] = self.interpolateInterior(self.uField)
            
            # Interpolate last point.
            field[:, -1] = (self.uField[:-1, 0] + self.uField[:-1, -1] + 
                            self.uField[1:, 0]  + self.uField[1:, -1])/4
        
        # If using reflective boundary conditions we only want interior points.
        elif not (self.periodicY or self.periodicX):
            return self.interpolateInterior(self.uField)
    
    def gradientFieldX(self, field):
        """ 
        """
        
        # Calculates field based on periodic boundary conditions.
        if self.periodicX:
            return (np.roll(field, -1, 1) - field)/self.dx
        
        # Calculates field based on reflective boundary conditions.
        return (field[:, 1:] - field[:, :-1])/self.dx
        
    def gradientFieldY(self, field):
        """ 
        """
        
        # Calculates field based on periodic boundary conditions.
        if self.periodicY:
            return (np.roll(field, -1, 0) - field)/self.dy
        
        # Calculates field based on reflective boundary conditions.
        return (field[1:, :] - field[:-1, :])/self.dy
        
    def interpolateInterior(self, phi):
        """ 
        """
        return (phi[:-1, :-1] + phi[1:, :-1] + phi[:-1, 1:] + phi[1:, 1:])/4