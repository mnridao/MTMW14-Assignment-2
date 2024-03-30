"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def forwardBackwardSchemeCoupled(funcs, grid, dt, nt):
    """ 
    
    Inputs
    -------
    
    Returns
    -------
    """
    # Forward-euler for scalar field (height perturbation).
    grid.fields[funcs[0].name] += dt*funcs[0](grid)
    
    # Iterate over velocity fields.
    for func in reversed(funcs[1:]) if nt%2 != 0 else funcs[1:]:
        
        # Forward-euler step for the current velocity field.
        grid.fields[func.name] += dt*func(grid)
    

def RK4SchemeCoupled(funcs, grid, dt, nt):
    """ 
    sad
    """
    
    # Initialise k as dict (please think of something better).
    kfunc = {}
    for func in funcs:
        kfunc[func.name] = np.zeros(shape=(4, *grid.fields[func.name].shape))
            
    gridP = grid
    for i in range(4):
        for func in funcs:
                        
            # Find k_i for the current function.
            kfunc[func.name][i] = func(gridP)
        
        if i==3: break # break out early to avoid mess below.
        
        # Make a copy to reset fields to previous timestep.
        gridP = grid.copy()  #oof
        
        # Update the intermediate prediction grid (another loop!).
        for func in funcs:
                        
            # Update the prediction grid for the next prediction.
            gridP.fields[func.name] += dt*kfunc[func.name][i]*(0.5 if i < 2 else 1)
            
    # Runge-kutte step (loop number 4!).
    for func in funcs:
        
        k = kfunc[func.name]
        grid.fields[func.name] += dt*(k[0] + 2*k[1] + 2*k[2] + k[3])/6

def semiImplicitSchemeCoupled(self, funcs, grid, dt, nt): 
    """ 
    """
    
    # Setup matrix equation to solve for current time step.
    

class SemiLagrangianSchemeCoupled:
    """ 
    """
    
    def __init__(self, interpMethod="linear"):
        """ 
        """
        self.interpMethod = interpMethod
                
        # Store previous time step wind (think of better way?).
        self.uFieldOld = None 
        self.vFieldOld = None
        
    def __call__(self, funcs, grid, dt, nt):
        """ 
        """            
        # Keep a copy of the original grid (won't be updated).
        gridOld = grid.copy()
                        
        # Update the height perturbation field first.
        self.updateField(grid, gridOld, funcs[0], dt)
        
        # Iterate over velocity fields.
        for func in reversed(funcs[1:]) if nt%2 != 0 else funcs[1:]:
            
            # Find departure point.
            self.updateField(grid, gridOld, func, dt)
        
        # Store previous time step for next round (used in dp calc).
        self.uFieldOld = gridOld.fields["uVelocity"]
        self.vFieldOld = gridOld.fields["vVelocity"]
            
    def updateField(self, grid, gridOld, func, dt):
        """ 
        """
        
        # Calculate departure point.
        Xdp, Ydp = self.calculateDeparturePoint(gridOld, dt, func.name)
                
        # Departure point forcing calculations should be done at gridOld but blows up.
        # gridDP = gridOld
        gridDP = grid
        if func.name == "eta":
            
            fieldDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Ymid[:, 0], 
                                       gridDP.fields[func.name])
            forcingsDP = self.calculateEtaForcings(Xdp, Ydp, func, gridDP)
            
        elif func.name == "uVelocity":
            
            fieldDP = self.interpolate(Xdp, Ydp, grid.X[0, 1:-1], grid.Ymid[:, 0],
                                       gridDP.fields[func.name])
            forcingsDP = self.calculateUForcings(Xdp, Ydp, fieldDP, func, gridDP)

        elif func.name == "vVelocity":
            
            fieldDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Y[1:-1, 0], 
                                       gridDP.fields[func.name])  
            forcingsDP = self.calculateVForcings(Xdp, Ydp, fieldDP, func, gridDP)
        
        # Calculate forcings at current time step.
        forcingsNew = func(grid)
        
        # Update the current field ([:] required so that it doesn't break link).
        grid.fields[func.name][:] = (fieldDP + 0.5*dt*(forcingsNew + forcingsDP))
        
    def calculateDeparturePoint(self, gridOld, dt, funcName):
        """ 
        """
        
        # Find the appropriate grid and fields for the current func.
        if funcName == "eta":
            
            # Eta stored at half cell points.
            U = 0.5*(gridOld.uField[:, :-1] + gridOld.uField[:, 1:])
            V = 0.5*(gridOld.vField[:-1, :] + gridOld.vField[1:, :])
            
            X = gridOld.Xmid 
            Y = gridOld.Ymid
            
        elif funcName == "uVelocity":
            
            # u-velocity stored at half cells in y and full cells in x.
            U = gridOld.uField[:, 1:-1]
            V = gridOld.vOnUField()
            
            X = gridOld.X[:-1, 1:-1]
            Y = gridOld.Ymid[:, :-1]
            
        elif funcName == "vVelocity":
            
            # v-velocity stored at half cells in x and full cells in y.
            U = gridOld.uOnVField()
            V = gridOld.vField[1:-1, :]
            
            X = gridOld.Xmid[:-1, :]
            Y = gridOld.Y[1:-1, :-1]
        
        # Intermediate departure point.
        Xstar = X - 0.5*dt*U
        Ystar = Y - 0.5*dt*V
                
        # Find velocities at intermediate departure point.
        Ustar = self.interpolate(Xstar, Ystar, gridOld.X[0, 1:-1], gridOld.Ymid[:, 0], 
                                 gridOld.uField[:, 1:-1])
        Vstar = self.interpolate(Xstar, Ystar, gridOld.Xmid[0, :], gridOld.Y[1:-1, 0], 
                                 gridOld.vField[1:-1, :])
        
        if np.any(self.uFieldOld):
            
            UstarOld = self.interpolate(Xstar, Ystar, gridOld.X[0, 1:-1], 
                                        gridOld.Ymid[:, 0], self.uFieldOld)
            VstarOld = self.interpolate(Xstar, Ystar, gridOld.Xmid[0, :], 
                                        gridOld.Y[1:-1, 0], self.vFieldOld)
            
            Ustar = 1.5*Ustar - 0.5*UstarOld
            Vstar = 1.5*Vstar - 0.5*VstarOld
            
        Xdp = X - dt*Ustar 
        Ydp = Y - dt*Vstar
                
        return Xdp, Ydp
    
    def calculateEtaForcings(self, Xdp, Ydp, func, grid):
        """ 
        """
        
        # Forcing arguments for equation.
        dudxDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Ymid[:, 0], 
                                  grid.dudxField())
        dvdyDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Ymid[:, 0],
                                  grid.dvdyField())
        
        return func.forcings(dudxDP, dvdyDP)
        
    def calculateUForcings(self, Xdp, Ydp, fieldDP, func, grid):
        """ 
        """
        
        # Forcing arguments for equation.
        detadxDP = self.interpolate(Xdp, Ydp, grid.X[0, 1:-1], grid.Ymid[:, 0],
                                    grid.detadxField())
        vFieldDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Y[1:-1, 0],
                                    grid.fields["vVelocity"])


        return func.forcings(fieldDP, vFieldDP, detadxDP, grid.Ymid, grid.xbounds[1]) 
                
    def calculateVForcings(self, Xdp, Ydp, fieldDP, func, grid):
        """ 
        """
        
        # Forcing arguments for equations.
        detadyDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Y[1:-1, 0], 
                                    grid.detadyField())
        uFieldDP = self.interpolate(Xdp, Ydp, grid.X[0, 1:-1], grid.Ymid[:, 0], 
                                    grid.fields["uVelocity"])
        
        return func.forcings(uFieldDP, fieldDP, detadyDP, grid.Y[1:-1, :-1])
    
    def interpolate(self, Xdp, Ydp, Xgrid, Ygrid, field):
        """ 
        Worth considering?
        """
        
        interp = RegularGridInterpolator((Ygrid, Xgrid), field, bounds_error=False, 
                                         fill_value=None, 
                                         method=self.interpMethod)
        
        return interp((Ydp, Xdp))

# TODO: Semi implicit
# TODO: Semi implicit semi lagrangian?

class SemiImplicitSemiLagrangianCoupled(SemiLagrangianSchemeCoupled):
    
    def __init__(self):
        super().__init__()
        
    def updateField(self, grid, gridOld, func, dt):
        """ 
        Can I overwrite this function?
        """

def RK4SchemeCoupled_OLD(funcs, grid, dt, nt):
    """ 
    might be better.
    """
    
    # hmm
    k1 = funcs[0](grid)
    l1 = funcs[1](grid)
    m1 = funcs[2](grid)
    
    ## SECOND STAGE ##
    gridP = grid.copy()
    
    gridP.fields["eta"] += 0.5*dt*k1 
    gridP.fields["uVelocity"] += 0.5*dt*l1
    gridP.fields["vVelocity"] += 0.5*dt*m1 
    
    k2 = funcs[0](gridP)
    l2 = funcs[1](gridP)
    m2 = funcs[2](gridP)
    
    ## THIRD STAGE ##
    gridP = grid.copy()
    
    gridP.fields["eta"] += 0.5*dt*k2 
    gridP.fields["uVelocity"] += 0.5*dt*l2
    gridP.fields["vVelocity"] += 0.5*dt*m2
    
    k3 = funcs[0](gridP)
    l3 = funcs[1](gridP)
    m3 = funcs[2](gridP)
    
    ## FOURTH STAGE ##
    gridP = grid.copy()
    
    gridP.fields["eta"] += dt*k3 
    gridP.fields["uVelocity"] += dt*l3 
    gridP.fields["vVelocity"] += dt*m3 
    
    k4 = funcs[0](gridP)
    l4 = funcs[1](gridP)
    m4 = funcs[2](gridP)
    
    ## Runge-Kutte step.
    grid.fields["eta"] += dt*(k1 + 2*k2 + 2*k3 + k4)/6
    grid.fields["uVelocity"] += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    grid.fields["vVelocity"] += dt*(m1 + 2*m2 + 2*m3 + m4)/6