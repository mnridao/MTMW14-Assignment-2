"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
    
class ArakawaCGrid:
    """
    Class that contains the information stored on the Arakawa-C grid. Also 
    responsible for manipulating information stored on the grid (maybe it 
    shouldn't be).
    """
    
    def __init__(self, xbounds, nx, ybounds=None, ny=None, periodicX=False):
        """ 
        Inputs
        -------
        xbounds   : list 
                    Contains the lower and upper bounds of the x-domain.
        nx        : int 
                    Number of grid points in the x direction.
        ybounds   : list or None 
                    Contains the lower and upper bounds of the y-domain. If 
                    this is None, it is set equal to xbounds.
        ny        : int or None 
                    Number of grid points in the y direction. If this is None, 
                    it is set equal to nx.
        periodicX : bool or None
                    Specifies whether periodic boundary conditions should be 
                    set in the x-direction. Default is False.
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
        Makes a copy of the grid. I don't really like this, but I have made 
        some questionable decisions regarding the implementation of my 
        semi-lagrangian and semi-implicit schemes.
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
        Sets up the full and half grid points of the Arakawa-C grid.
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
        Resets the fields of the prognostic variables to zero, allowing the 
        grid to be used again in the solver.
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
        Returns the grid points used by the u-velocity field.
        """
        return (self.X[:-1, :], np.hstack((self.Ymid, self.Ymid[:, 0].reshape(-1, 1))))
        
    def vGrid(self):
        """ 
        Returns the grid points used by the v-velocity field.
        """
        return (np.vstack((self.Xmid, self.Xmid[0, :].reshape(1, -1))), self.Y[:, :-1])
        
    def etaGrid(self):
        """ 
        Returns the grid points used by the eta field.
        """
        return (self.Xmid, self.Ymid)
    
    def dudxField(self):
        """ 
        Returns the gradient of the u-velocity field in the x-direction. This 
        is on the eta grid.
        """
        return self.forwardGradientFieldX(self.uField)
            
    def dvdyField(self):
        """
        Returns the gradient of the v-velocity field in the y-direction. This 
        is on the eta grid.
        """
        return self.forwardGradientFieldY(self.vField)
        
    def detadxField(self):
        """
        Returns the gradient of the eta field in the x direction. This is on 
        the full grid points in the x direction, and half grid points in the
        y direction.
        """
        return self.backwardGradientFieldX(self.hField)
    
    def detadyField(self):
        """ 
        Returns the gradient of the eta field in the y direction. This is on 
        the half grid points in the x direction, and full grid points in the
        y direction.
        """
        return self.backwardGradientFieldY(self.hField)
    
    def vorticityField(self):
        """ 
        Returns the vorticity field. Thought I would have made some Rossby 
        waves by now :(
        """
        return (self.forwardGradientFieldX(self.vField) - 
                self.forwardGradientFieldY(self.uField))
    
    def uOnEtaField(self):
        """ 
        Returns the u-velocity field interpolated onto the eta grid.
        """
        return 0.5*(self.uField[:, :-1] + self.uField[:, 1:])
        
    def vOnEtaField(self):
        """ 
        Returns the v-velocity field interpolated onto the eta grid.
        """
        return 0.5*(self.vField[:-1, :] + self.vField[1:, :])
    
    def vOnUField(self):
        """ 
        Returns the v-velocity field interpolated onto the u-grid.
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
        Returns the u-velocity field interpolated onto the v-grid.
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
        Calculates the forward gradient of field in the x-direction.
        """
        
        # Calculates field based on periodic boundary conditions.
        if self.periodicX:
            return (np.roll(field, -1, 1) - field)/self.dx
        
        # Calculates field based on reflective boundary conditions.
        return (field[:, 1:] - field[:, :-1])/self.dx
    
    def backwardGradientFieldX(self, field):
        """ 
        Calculates the backward gradient of field in the x-direction. There is 
        a lot of repetition here, I got very tired and never came back here. I 
        am rediscovering these functions now that I am having to comment them.
        """
        
        # Calculates field based on periodic boundary conditions.
        if self.periodicX:
            return (field - np.roll(field, 1, 1))/self.dx
        
        # Calculates field based on reflective boundary conditions. (same?)
        return (field[:, 1:] - field[:, :-1])/self.dx # i think ok
    
    def forwardGradientFieldY(self, field):
        """ 
        Calculates the forward gradient of field in the y-direction.
        """
        
        # Calculates field based on periodic boundary conditions.
        if self.periodicY:
            return (np.roll(field, -1, 0) - field)/self.dy
        
        # Calculates field based on reflective boundary conditions.
        return (field[1:, :] - field[:-1, :])/self.dy
    
    def backwardGradientFieldY(self, field):
        """ 
        Calculates the backward gradient of field in the y-direction.
        """
        # Calculates field based on periodic boundary conditions.
        if self.periodicY:
            return (field - np.roll(field, 1, 0))/self.dy
        
        # Calculates field based on reflective boundary conditions. (same?)
        return (field[1:, :] - field[:-1, :])/self.dy # i think ok
    
    def interpolateInterior(self, phi):
        """ 
        Interpolates the interior points.
        """
        return (phi[:-1, :-1] + phi[1:, :-1] + phi[:-1, 1:] + phi[1:, 1:])/4