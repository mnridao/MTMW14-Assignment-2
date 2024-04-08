"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
from scipy.stats import multivariate_normal

class Model:
    """ 
    Class responsible for holding the equations to the SWE problem and the 
    grid. Problem parameters and initial conditions are set through this class.
    """
        
    def __init__(self, eqns, grid):
        """ 
        Inputs
        -------
        eqns : list of BaseEqnSWE objects
               Each of the coupled equations that appears in the problem.
       grid  : ArakawaCGrid object
               Object containing the domain and state information for the 
               problem.
        """
        # Model equations and domain.
        self.eqns = eqns
        self.grid = grid
    
    def setGamma(self, gamma):
        """ 
        Set the magnutide of the drag coefficient.
        """
        for eqn in self.eqns:
            eqn.params.gamma = gamma
    
    def setTau0(self, tau0):
        """ 
        Set the amplitude of the wind stress for the SWE problem.
        
        Inputs
        ------
        tau0 : float
               Wind stress amplitude.
        """
        
        # Update the wind stress amplitude for equations in the model.
        for eqn in self.eqns:
            eqn.params.tau0 = tau0
    
    def setH(self, H):
        """ 
        Set the height of the background state for the SWE problem.
        
        Inputs
        ------
        H : float
            Reference height.
        """
        
        # Update H for each of the equations in the model.
        for eqn in self.eqns:
            eqn.params.H = H
                
    def setf0(self, f0):
        """ 
        Set the coriolis parameter at a reference latitude for the SWE problem.
        
        Inputs
        ------
        fo : float
             Coriolis parameter at a reference latitude.
        """
        # Update f0 for each of the equations in the model.
        for eqn in self.eqns:
            eqn.params.f0 = f0
    
    def setBeta(self, beta):
        """ 
        Set the gradient of the coriolis parameter with respect to latitude.
        
        Inputs
        ------
        beta : float
               The gradient of the coriolis parameter with resepct to latitude.
        """
        # Update beta for each of the equations in the model.
        for eqn in self.eqns:
            eqn.params.beta = beta
        
    def activateWindStress(self, activate):
        """
        Activate the wind stress.
        
        Inputs
        ------
        activate : bool
                   Turn wind forcing on/off.
        """
        # Activate or deactivate the wind stress for all the model equations.
        for eqn in self.eqns:
            eqn.params.setWindStressX("default" if activate else "off")
            eqn.params.setWindStressY("default" if activate else "off")
    
    def activateDamping(self, activate, gamma=None):
        """
        Activate the damping.
        
        Inputs
        ------
        activate : bool
                   Turn damping on/off.
        gamma    : Value of gamma that will be set if damping is turned on, 
                   default value is restored if gamma=None.
        """
        
        # Activate or deactivate the damping for all the model equations.
        for eqn in self.eqns:
            eqn.params.activateDamping(activate, gamma)
    
    def setBlobInitialCondition(self, mu, var, height):
        """ 
        Initialises the height perturbation field (eta) with a 2D Gaussian 
        blob at a specified location.
        
        Inputs
        -------
        mu     : np array
                 Array of the mean location of the 2D Gaussian blob.
        var    : np array
                 Array of the variance of the 2D Gaussian blob. Or is it 
                 standard deviation? I've forgotten. Basically the spread.
        height : float
                 Height of the maximum peak of the Gaussian blob.
        """
        
        self.grid.hField += self.createBlob(mu, var, height)
        
        # Update hField view - this is stupid.
        self.grid.fields["eta"] = self.grid.hField
    
    def setMountainBottomTopography(self, mu, var, height):
        """ 
        Initialises the bottom topography of the problem. Default is flat. 
        
        I am not convinced that this works.
        
        Inputs
        -------
        mu     : np array
                 Array of the mean location of the 2D Gaussian blob.
        var    : np array
                 The spread of the 2D Gaussian blob.
        height : float
                 Height of the maximum peak of the Gaussian blob.
        """
        
        self.grid.hBot += self.createBlob(mu, var, height)
        
    def createBlob(self, mu, var, height):
        """ 
        Creates the 2D Gaussian blob.
        
        Inputs
        -------
        mu     : np array
                 Array of the mean location of the 2D Gaussian blob.
        var    : np array
                 The spread of the 2D Gaussian blob.
        height : float
                 Height of the maximum peak of the Gaussian blob.
        """
        # Create the Gaussian blob.
        pos = np.empty(self.grid.Xmid.shape + (2,))
        pos[..., 0] = self.grid.Xmid
        pos[..., 1] = self.grid.Ymid        
        pdf = multivariate_normal(mu, [[var[0], 0], [0, var[1]]]).pdf(pos)
    
        return pdf * (height/pdf.max())    
    
    def setSharpShearInitialCondition(self, Vmean=50, f0=1e-4, beta=1.6e-11):
        """ 
        This is from when I was trying to make Rossby waves. Inspired by 
        Robin Hogan.
        
        I don't think that this works.
        """
        
        # Set parameters for the initial condition.
        self.setf0(f0)
        self.setBeta(beta)
        
        # Deactivate wind stress just in case.
        self.activateWindStress(False)
        
        # Sharp shear height initial condition from Robin Hogan.
        
        height = ((Vmean*self.eqns[0].params.f0/self.eqns[0].params.g)
                  *np.abs(self.grid.Ymid - self.grid.Ymid.mean()))
        self.grid.hField = height - height.mean()
        
        # Add random noise.
        self.addRandomNoise()
        
        # Update viewer for height perturbation.
        self.grid.fields["eta"] = self.grid.hField
        
        # Initialise geostrophic balance for velocities. 
        self.setGeostrophicBalanceInitialCondition()
        
    def setGeostrophicBalanceInitialCondition(self):
        """ 
        This is from when I was trying to make Rossby waves. Inspired by 
        Robin Hogan.
        
        I don't think that this works.
        """
        
        # Parameters for calculation.
        Vmax = 200
        g = self.eqns[0].params.g 
        beta = self.eqns[0].params.beta
        f0 = self.eqns[0].params.f0
        
        ### Geostrophic balance for v - only works if periodic in X ###
        detadx = 0.5*(np.roll(self.grid.hField[:-1, :], -1, 1) - 
                      np.roll(self.grid.hField[:-1, :], 1, 1))/self.grid.dx
        
        self.grid.vField[1:-1, :] = (g/(f0 + beta*(self.grid.Y[1:-1, :] - 
                                                   self.grid.Y[1:-1, :].mean()))*
                                     detadx)

        # Meridional velocity zero at horizontal boundaries.
        self.grid.vField[0, :]  = 0.
        self.grid.vField[-1, :] = 0.
        
        # Filter out velocities greater than the max.
        self.grid.vField[self.grid.vField > Vmax] = Vmax
        self.grid.vField[self.grid.vField < -Vmax] = -Vmax
        
        # Update viewer.
        self.grid.fields["vVelocity"] = self.grid.vField[1:-1, :]
        
        ### Geostrophic balance for u ###

        detady = 0.5*(self.grid.hField[2:, :]-self.grid.hField[:-2, :])/self.grid.dy

        self.grid.uField[1:-1, :] = (- g/(f0 + beta*(self.grid.Ymid - 
                                                     self.grid.Ymid.mean()))[1:-1, :]
                                     *detady)
            
        self.grid.uField[[0, -1], :] = self.grid.uField[[1, -2], :]
                
        self.grid.uField[self.grid.uField < -Vmax] = -Vmax
        self.grid.uField[self.grid.uField > Vmax] = Vmax
        
        self.grid.fields["uVelocity"] = self.grid.uField
        
    def addRandomNoise(self):
        """ 
        This is from when I was trying to make Rossby waves. Inspired by 
        Robin Hogan.
        
        I don't think that this works.
        """
        
        # Coriolis plane.
        F = (self.eqns[0].params.f0 + self.eqns[0].params.beta*self.grid.Ymid)
        
        r, c = self.grid.hField.shape
        self.grid.hField += 1*np.random.randn(r, c)*(F/self.eqns[0].params.f0)