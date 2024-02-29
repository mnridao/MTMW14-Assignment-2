"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

class State(object):
    """ 
    Contains and manages the variables corresponding the current state."""
    
    def __init__(self):
        
        self.uField = None 
        self.vField = None 
        self.hField = None # Maybe just make field that can hold different scalars.

class ArakawaCGrid(object):
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
        
        # Set up arrays representing the 2D grid.
        self.createGrid()
        
        # Initialise state variables.
        self.initialiseState()
        
    def createGrid(self):
        """ 
        """
        xpoints = np.linspace(self.xbounds[0], self.xbounds[1], self.nx+1)
        ypoints = np.linspace(self.ybounds[0], self.ybounds[1], self.ny+1)
        
        self.X, self.Y = np.meshgrid(xpoints, ypoints)
    
    def initialiseState(self):
        """ 
        """
        self.state = State()
        self.state.uField = np.zeros(shape=(self.ny, self.nx+1))
        self.state.vField = np.zeros(shape=(self.ny+1, self.nx))
        self.state.hField = np.zeros(shape=(self.nx, self.ny))