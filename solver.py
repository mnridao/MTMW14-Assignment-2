"""
MTMW14 Assignment 2

Student ID: 31827379
"""
import numpy as np

from plotters import plotContourSubplot
    
class Solver:
    """ Class that runs the simulaton."""
    
    def __init__(self, model, scheme, dt, nt, store=False):
        """ 
        Object initialisation.
        
        model  : Model object
                 Class that contains the equations for the current problem.
        scheme : callable object
                 The time scheme used in the simulation.
        dt     : float
                 Timestep.
        nt     : int 
                 Number of time steps to run in the simulation.
        store  : bool 
                 Turn storage on/off. Only really used for animations.
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
        Run the solver.
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
                
            # Store state if necessary (should probably store in arrays).
            if self.store and t % self.storeEveryN == 0:
                self.history.append([self.model.grid.uField.copy(),
                                     self.model.grid.vField.copy(),
                                     self.model.grid.hField.copy()])
                
            if self.plotEveryTimestep:
                plotContourSubplot(self.model.grid)
                
    def addCustomEquations(self, key, customEqn, nres=1):
        """ 
        Add an equation that will be evaluated at each timestep.
        
        Inputs
        -------
        key       : string
                    Key for the equation being added. This is for easy
                    accessing later.
        customEqn : callable object that takes a Model object as its argument.
                    Custom equation, e.g. to calculate energy, that will be
                    evaluated every timestep.
        nres      : int
                    Number of results returned from customEqn. Default is 1.
        """        
        # Initialise results data for custom function.
        data = np.zeros(shape=(self.nt+1, nres))
        data[0] = customEqn(self.model)
        
        # Store in dict for easy accessing.
        self.customEquations[key] = {"func": customEqn, "data": data}
        
    def getCustomData(self, key):
        """ 
        Getter for the data obtained from evaluating the custom equation 
        specified by key at every timestep.
        
        Inputs
        ------
        key : string
              Key for obtaining the data from the dictionary storing the 
              custom equation results.
        """
        return self.customEquations[key]["data"]
    
    def setNewTimestep(self, dt, endtime):
        """ 
        Set a new timestep for the problem. Set through this class for safety 
        incase there are any equations stored in customEquations dictionary - 
        this way the data arrays are re-initialised to the correct length.
        
        Inputs
        ------
        dt      : float
                  New timestep.
        endtime : float
                  Required endtime for the simulation. Used to recalculate nt.
        """
        self.dt = dt
        self.nt = int(np.ceil(endtime/dt))
        
        # Update customEquations data fields.
        for key, val in self.customEquations.items():
            
            # Change the size of the data array for the new nt.            
            self.addCustomEquations(key, val["func"], val["data"].shape[1])