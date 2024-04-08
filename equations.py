"""
MTMW14 Assignment 2

Student ID: 31827379
"""

from abc import ABC, abstractmethod
import numpy as np

class Parameters:
    """ 
    Class responsible for storing and updating the parameters of the SWE 
    problem.
    """
    
    def __init__(self):
        
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
        Sets the default wind stress for a subtropical gyre, i.e. easterly 
        trade winds in the equatorward half of the basin and westerlies in the 
        poleward half.
        """
        self.setWindStressX("default")
        self.setWindStressY("default")
    
    def turnOffWindStress(self):
        """ 
        Turns off the wind stress.
        """
        self.setWindStressX("off")
        self.setWindStressY("off")
    
    def setWindStressX(self, key, tau=None):
        """ 
        Set the wind stress according to a key value.
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
        Set the vertical component of the wind stress, default is None.
        """
        if key == "default" or key == "off":
            self.tauY = lambda Y: np.zeros_like(Y)
        
        elif key == "custom" and tau != None:
            self.tauY = tau
    
    def activateDamping(self, activate, gamma):
        """ 
        Turn the damping on or off.
        """
        if activate:
            self.gamma = gamma if gamma else 1e-6 # Reset to default if None.
        else:
            self.gamma = 0.
    
class BaseEqnSWE(ABC):
    """
    Base class for the SWEs.
    """
    
    def __init__(self):
        
        # Contains the default parameters for the SWE.
        self.params = Parameters()
    
    def __call__(self, grid):
        """
        This function is called to evaluate any BaseEqnSWE object.
        """        
        return self._f(grid)
    
    @abstractmethod
    def _f(self, grid):
        pass
    
    @abstractmethod
    def forcings(self):
        pass
        
    @abstractmethod 
    def explicitTerms(self, grid):
        pass
    
class UVelocity(BaseEqnSWE):
    """ 
    U-Velocity equation for the SWEs.
    """
    
    def __init__(self):
        super().__init__() 
        
        # Stored here so that we can iterate over functions in the schemes.
        self.name = "uVelocity"
            
    def _f(self, grid):
        """ 
        Calls this function when the equation must be evaluated.
        """        
        # Height perturbation gradient in x-direction.
        detadx = grid.detadxField()
        u = grid.fields["uVelocity"]
        v = grid.vOnUField()
            
        return self.forcings(u, v, detadx, grid.Ymid, grid.xbounds[1])
        
    def forcings(self, u, v, detadx, Y, L):
        """ 
        Returns the right-hand side of the u-momentum equation for the 
        shallow water equations. All inputs must be on the u-grid.
        """
        
        # Coriolis parameter (at half grid points - assumes c grid).
        f = (self.params.f0 + self.params.beta*Y)[..., :u.shape[1]]    
        
        # Wind forcing in x-direction.
        tauX = self.params.tauX(Y, L)[..., :u.shape[1]]
        
        return (f*v - self.params.g*detadx - 
                self.params.gamma*u + tauX/(self.params.rho*self.params.H))
    
    def explicitTerms(self, grid):
        """ 
        A lot of repetition here (easier atm) - will improve this if I have 
        time.
        """
        u = grid.fields["uVelocity"]
        
        # Coriolis parameter (at half grid points - assumes c grid).
        f = (self.params.f0 + self.params.beta*grid.Ymid)[..., :u.shape[1]]  
        
        # Wind forcing in x-direction.
        tauX = self.params.tauX(grid.Ymid, grid.xbounds[1])[..., :u.shape[1]]
        
        # Return the forcings without the gravity terms.
        return (f*grid.vOnUField() - self.params.gamma*u + 
                tauX/(self.params.rho*self.params.H))
    
class VVelocity(BaseEqnSWE):
    """ 
    V-Velocity equation for the SWEs.
    """
    
    def __init__(self):
        super().__init__()
        
        # Stored here so that we can iterate over functions in the schemes.
        self.name = "vVelocity"
            
    def _f(self, grid):
        """ 
        Calls this function when the equation must be evaluated.
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
        Returns the right-hand side of the v-momentum equation for the 
        shallow water equations. All inputs must be on the v-grid.
        """
        
        # Coriolis parameter.
        f = self.params.f0 + self.params.beta*Y
        
        # Wind forcing in y-direction.        
        tauY = self.params.tauY(Y)
                
        return (-f*u - self.params.g*detady - self.params.gamma*v + 
                tauY/(self.params.rho*self.params.H))
    
    def explicitTerms(self, grid):
        """ 
        A lot of repetition here (easier atm) - will improve this if I have 
        time.
        """
        
        # Get the v field.
        v = grid.fields["vVelocity"]
        
        # Values of Y for the current problem.
        Y = (grid.Y if grid.periodicY else grid.Y[1:-1, :])[..., :v.shape[1]]
        
        # Coriolis parameter.
        f = self.params.f0 + self.params.beta*Y
        
        # Wind forcing in y-direction.        
        tauY = self.params.tauY(Y)
        
        return (-f*grid.uOnVField() - self.params.gamma*v + 
                tauY/(self.params.rho*self.params.H))
    
class Eta(BaseEqnSWE):
    """ 
    Surface elevation equation for the SWEs.
    """

    def __init__(self):
        super().__init__()
        
        # Stored here so that we can iterate over functions in the schemes.
        self.name = "eta"
        
    def _f(self, grid):
        """ 
        Calls this function when the equation must be evaluated.
        """
                
        # Calculate new height perturbation.
        return self.forcings(grid.dudxField(), grid.dvdyField())
    
    def forcings(self, dudx, dvdy):
        """ 
        Returns the right-hand side of the continuity equation for the 
        shallow water equations. All inputs must be on the eta-grid.
        """
        return - self.params.H*(dudx + dvdy)

    def explicitTerms(self, grid):
        """ 
        None for my current implementation of semi-implicit.
        """
        pass