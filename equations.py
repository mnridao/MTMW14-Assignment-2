"""
MTMW14 Assignment 2

Student ID: 31827379
"""

from abc import ABC, abstractmethod
import numpy as np

class Parameters:
    """ 
    """
    
    def __init__(self):
        """ 
        """        
        
        ## DEFAULT PARAMETERS ##
        
        self.f0    = 1e-4  # Coriolis parameter [s^-1]
        self.beta  = 1e-11 # Coriolis parameter gradient [m^-1s^-1] 
        self.g     = 10    # Gravitational acceleration [ms^-2]
        self.gamma = 1e-6  # Linear drag coefficient [s^-1]
        self.rho   = 1000  # Density of water [kgm^-3]
        self.H     = 1000  # Height [m]

        ## DEFAULT FORCING PARAMETERS ##
        self.setDefaultWindStress()
                
    def setDefaultWindStress(self):
        """ 
        """
        self.setWindStressX("default")
        self.setWindStressY("default")
    
    def turnOffWindStress(self):
        """ 
        """
        self.setWindStressX("off")
        self.setWindStressY("off")
    
    def setWindStressX(self, key, tau=None):
        """ 
        """
        if key == "default":
            self.tau0 = 0.2 # Wind stress amplitude [Nm^-2]
            self.tauX = lambda Y, L: - self.tau0*np.cos(np.pi*Y/L)
            
        elif key == "off":
            self.tauX = lambda Y, L: np.zeros_like(Y)
        
        elif key == "custom" and tau != None:
            self.tauX = tau
        
    def setWindStressY(self, key, tau=None):
        """ 
        """
        if key == "default" or key == "off":
            self.tauY = lambda Y: np.zeros_like(Y)
        
        elif key == "custom" and tau != None:
            self.tauY = tau
        
class BaseEqnSWE(ABC):
    """ 
    """
    
    def __init__(self):
        
        # Contains the default parameters for the SWE.
        self.params = Parameters()
    
    def __call__(self, grid):
        """
        """
        # params.updateParams()
        
        return self._f(grid)
    
    @abstractmethod
    def _f(self, grid):
        pass
    
    @abstractmethod
    def forcings(self):
        pass
        
class UVelocity(BaseEqnSWE):
    """ 
    """
    
    def __init__(self):
        super().__init__() 
        
        self.name = "uVelocity"
            
    def _f(self, grid):
        """ 
        """        
        # Height perturbation gradient in x-direction.
        detadx = grid.detadxField()
        u = grid.fields["uVelocity"]
        v = grid.vOnUField()
            
        return self.forcings(u, v, detadx, grid.Ymid, grid.xbounds[1])
        
    def forcings(self, u, v, detadx, Y, L):
        """ 
        """
        
        # Coriolis parameter (at half grid points - assumes c grid).
        f = (self.params.f0 + self.params.beta*Y)[..., :u.shape[1]]
        # f = (self.params.f0 + self.params.beta*(Y - Y.mean()))[..., :u.shape[1]]        
        
        # Wind forcing in x-direction.
        tauX = self.params.tauX(Y, L)[..., :u.shape[1]]
        
        return (f*v - self.params.g*detadx - 
                self.params.gamma*u + tauX/(self.params.rho*self.params.H))
    
class VVelocity(BaseEqnSWE):
    """ 
    """
    
    def __init__(self):
        super().__init__()
        
        self.name = "vVelocity"
            
    def _f(self, grid):
        """ 
        """        
        
        # Height perturbation gradient in y-direction.
        detady = grid.detadyField()
        v = grid.fields["vVelocity"]
        u = grid.uOnVField()
        
        # Values of Y for the current problem.
        Y = (grid.Y if grid.periodicY else grid.Y[1:-1, :])[..., :v.shape[1]]
                
        return self.forcings(u, v, detady, Y)
        
    def forcings(self, u, v, detady, Y):
        """ 
        """
        
        # Coriolis parameter.
        f = self.params.f0 + self.params.beta*Y
        # f = self.params.f0 + self.params.beta*(Y - Y.mean())
        
        # Wind forcing in y-direction.        
        tauY = self.params.tauY(Y)
                
        return (-f*u - self.params.g*detady - self.params.gamma*v + 
                tauY/(self.params.rho*self.params.H))
    
class Eta(BaseEqnSWE):
    """ 
    """

    def __init__(self):
        super().__init__()
        
        self.name = "eta"
        
    def _f(self, grid):
        """ 
        """
                
        # Calculate new height perturbation.
        return self.forcings(grid.dudxField(), grid.dvdyField())
    
    def forcings(self, dudx, dvdy):
        """ 
        """
        return - self.params.H*(dudx + dvdy)