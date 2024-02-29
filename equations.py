"""
MTMW14 Assignment 2

Student ID: 31827379
"""

class Parameters(object):
    """ 
    """
    
    def __init__(self):
        
        self.isWindStressOn = True
        
        ## DEFAULT PARAMETERS ##
        
        self.f0    = 1e-4  # Coriolis parameter [s^-1]
        self.beta  = 1e-11 # Coriolis parameter gradient [m^-1s^-1] 
        self.g     = 10    # Gravitational acceleration [ms^-2]
        self.gamma = 1e-6  # Linear drag coefficient [s^-1]
        self.rho   = 1000  # Density of water [kgm^-3]
        self.H     = 1000  # Height [m]

        ## FORCING PARAMETERS ##
        
        self.tau0 = 0.2   # Wind stress amplitude [Nm^-2]
        
class BaseEqnSWE(object):
    """ 
    """
    
    def __init__(self):
        
        # Contains the default parameters for the SWE.
        self.params = Parameters()
    
    def __call__(self, u, v, eta):
        """ 
        Something like this...
        
        pass in time parameters if needed."""
        
        # Update parameters if neccesary.
        # params.updateParams() or smt
        
        return self._f(u, v, eta)
    
    def _f(self, u, v, eta):
        pass
    
class UVelocity(BaseEqnSWE):
    """ 
    """
    
    def __init__(self):
        super().__init__() 
        
    def _f(self, u, v, eta):
        
        pass
        
class VVelocity(BaseEqnSWE):
    """ 
    """
    
    def __init__(self):
        super().__init__()
        
    def _f(self, u, v, eta):
        pass
        
class Eta(BaseEqnSWE):
    """ 
    """
    
    def __init__(self):
        super().__init__()
        
    def _f(self, u, v, eta):
        pass