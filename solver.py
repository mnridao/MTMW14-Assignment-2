"""
MTMW14 Assignment 2

Student ID: 31827379
"""
import numpy as np
from scipy.stats import multivariate_normal

from plotters import plotContourSubplot
import matplotlib.pyplot as plt

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
        
        # Functions that should be called each iteration stored here.
        self.customEquations = {}
        
    def run(self, *phi0):
        """ 
        """
        # Reset the grid fields.
        self.model.grid.resetFields()
        
        # Run the simulation for each time step.
        for t in range(self.nt):
            
            # Update the grid (no return since it was passed by reference).
            self.scheme(self.model.eqns, self.model.grid, self.dt, t)
            
            # Store state if necessary.
            # ...
            
            # Evaluate any functions added by user (e.g. energy)
            for eqn in self.customEquations.values():
                
                # Evaluate the custom eqn for the current grid state.
                eqn["data"][t+1] = eqn["func"](self.model)
            
            # # plotContourSubplot(self.model.grid)
            # fig, ax = plt.subplots(figsize = (8, 8), facecolor = "white")
            # plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 19)
            # plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
            # plt.ylabel("y [km]", fontname = "serif", fontsize = 16)
            # q_int = 3
            # Q = ax.quiver(self.model.grid.X[::q_int, ::q_int]/1000.0, self.model.grid.Y[::q_int, ::q_int]/1000.0, self.model.grid.uField[::q_int,::q_int], self.model.grid.vField[::q_int,::q_int],
            #     scale=0.05, scale_units='inches')
            # plt.show()
    
    def runEnsemble(self, numEnsembles, perturbationRange, *phi0):
        """ 
        """
        pass
    
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