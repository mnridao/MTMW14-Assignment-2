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
        
        # self.windStressActivated = True # Not used atm.
        self.betaPlaneActive = True
        
        ## DEFAULT PARAMETERS ##
        
        self.f0    = 1e-4  # Coriolis parameter [s^-1]
        self.beta  = 1e-11 # Coriolis parameter gradient [m^-1s^-1] 
        self.g     = 10    # Gravitational acceleration [ms^-2]
        self.gamma = 1e-6  # Linear drag coefficient [s^-1]
        self.rho   = 1000  # Density of water [kgm^-3]
        self.H     = 1000  # Height [m]

        ## DEFAULT FORCING PARAMETERS ##
        self.setDefaultWindStress()
        
    def updateParameters(self):
        pass
        
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
    def _f(self, state):
        pass
        
class UVelocity(BaseEqnSWE):
    """ 
    """
    
    def __init__(self):
        super().__init__() 
        
        self.name = "uVelocity"
    
    # def _f(self, grid):
    #     """ 
    #     """
        
    #     # Height perturbation in x-direction.
    #     detadx = (np.roll(grid.hField, -1, 1) - grid.hField) /grid.dx
        
    #     # Coriolis parameter.
        
    def _f(self, grid):
        """ 
        """        
        # Height perturbation gradient in x-direction.
        # detadx = (grid.hField[:, 1:] - grid.hField[:, :-1])/grid.dx
        detadx = grid.detadxField()

        # Coriolis parameter (at half grid points assuming c grid - make more general?).
        f = (self.params.betaPlaneActive*
              (self.params.f0 + self.params.beta*grid.Ymid))
        
        # Wind forcing in x-direction.
        tauX = self.params.tauX(grid.Y, grid.xbounds[1])
        
        return (f[:, :-1]*grid.vOnUField() - self.params.g*detadx - 
                self.params.gamma*grid.uField[:, 1:-1] + 
                tauX[:-1, 1:-1]/(self.params.rho*self.params.H))
        
    def forcings(self, u, v, detadx, Y):
        """ 
        """
        
        pass
    
class VVelocity(BaseEqnSWE):
    """ 
    """
    
    def __init__(self):
        super().__init__()
        
        self.name = "vVelocity"
    
    # def _f(self, grid):
    #     """ 
    #     """
        
    #     pass
    
    def _f(self, grid):
        """ 
        """        
        # Height perturbation gradient in y-direction.
        # detady = (grid.hField[1:, :] - grid.hField[:-1, :])/grid.dy
        detady = grid.detadyField()

        # Coriolis parameter.
        f = (self.params.betaPlaneActive*
              (self.params.f0 + self.params.beta*grid.Y))
        
        # Wind forcing in y-direction.
        tauY = self.params.tauY(grid.Y)
                
        return (-f[1:-1,:-1]*grid.uOnVField() - self.params.g*detady - 
                self.params.gamma*grid.vField[1:-1, :] + 
                tauY[1:-1,:-1]/(self.params.rho*self.params.H))
        
    def forcings(self, u, v, detady, Y):
        """ 
        """
        
        # Calculate coriolis parameter on the grid.
        f = (self.params.betaPlaneActive*
             (self.params.f0 + self.params.beta*Y))
        
        # Calculate the wind forcing on the grid.
        tauY = self.params.tauY(Y)
        
        # TODO: can i make this more general for different bc?
        return (-f[1:-1,:-1]*u - self.params.g*detady - 
                self.params.gamma*v[1:-1, :] + 
                tauY[1:-1,:-1]/(self.params.rho*self.params.H))
    
class Eta(BaseEqnSWE):
    """ 
    """

    def __init__(self):
        super().__init__()
        
        self.name = "eta"
    
    # def _f(self, grid):
    #     """ 
    #     """
        
    #     pass
    
    def _f(self, grid):
        """ 
        """
        # Calculate velocity gradients throughout the domain.
        # dudx = (grid.uField[:, 1:] - grid.uField[:, :-1]) / grid.dx
        # dvdy = (grid.vField[1:, :] - grid.vField[:-1, :]) / grid.dy
                
        # Calculate new height perturbation.
        return self.forcings(grid.dudxField(), grid.dvdyField())
    
    def forcings(self, dudx, dvdy):
        """ 
        """
        return - self.params.H*(dudx + dvdy)