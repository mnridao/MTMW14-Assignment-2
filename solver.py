"""
MTMW14 Assignment 2

Student ID: 31827379
"""
import numpy as np

from plotters import plotContourSubplot
    
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
        self.storeEveryN  = 1
        
        # Functions that should be called each iteration stored here.
        self.customEquations = {}
        
        # Debugging tools.
        self.plotEveryTimestep = False
        
    def run(self):
        """ 
        """        
        # Initialise storage arrays if necessary (only really for animations).
        if self.store:
            self.history = []
        
        # Run the simulation for each time step.
        for t in range(self.nt):

            # Update the grid (no return since it was passed by reference).
            self.scheme(self.model.eqns, self.model.grid, self.dt, t)
                        
            # Evaluate any functions added by user (e.g. energy)
            for eqn in self.customEquations.values():
                eqn["data"][t+1] = eqn["func"](self.model)
                
            # Store state if necessary (Could just use grid.fields instead).
            if self.store and t % self.storeEveryN == 0:
                self.history.append([self.model.grid.uField.copy(),
                                     self.model.grid.vField.copy(),
                                     self.model.grid.hField.copy()])
            
            if self.plotEveryTimestep:
                plotContourSubplot(self.model.grid)
                
    def addCustomEquations(self, key, customEqn, nres=1):
        """ 
        """        
        # Initialise results data for custom function.
        data = np.zeros(shape=(self.nt+1, nres))
        data[0] = customEqn(self.model)
        
        # Store in dict for easy accessing.
        self.customEquations[key] = {"func": customEqn, "data": data}
        
    def getCustomData(self, key):
        """ 
        """
        return self.customEquations[key]["data"]
    
    def setNewTimestep(self, dt, endtime):
        """ 
        """
        self.dt = dt
        self.nt = int(np.ceil(endtime/dt))
        
        # Update customEquations data fields.
        for key, val in self.customEquations.items():
            
            # Change the size of the data array for the new nt.            
            self.addCustomEquations(key, val["func"], val["data"].shape[1])