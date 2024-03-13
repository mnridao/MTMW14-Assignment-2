"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
from scipy.stats import multivariate_normal

class Model:
    """ 
    """
        
    def __init__(self, eqns, grid):
        """ 
        """
        # Model equations and domain.
        self.eqns = eqns
        self.grid = grid
    
    def setH(self, H):
        """ 
        """
        
        # Update H for each of the equations in the model.
        for eqn in self.eqns:
            eqn.params.H = H
                
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
        
    def activateWindStress(self, activate):
        """
        """
        # Activate or deactivate the wind stress for all the model equations.
        for eqn in self.eqns:
            eqn.params.setWindStressX("default" if activate else "off")
            eqn.params.setWindStressY("default" if activate else "off")
            
    def setBlobInitialCondition(self, mu, var, height):
        """ 
        """
        # Create the Gaussian blob.
        pos = np.empty(self.grid.Xmid.shape + (2,))
        pos[..., 0] = self.grid.Xmid
        pos[..., 1] = self.grid.Ymid        
        pdf = multivariate_normal(mu, [[var[0], 0], [0, var[1]]]).pdf(pos)
                
        # Generate the blob height perturbation field.
        self.grid.hField += (height/pdf.max() * pdf)
        
        # Update hField view - this is stupid.
        self.grid.fields["eta"] = self.grid.hField
    
    def setSharpShearInitialCondition(self, Vmean=50, f0=1e-4, beta=1.6e-11):
        """ 
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
        """
        
        # Coriolis plane.
        # F = (self.eqns[0].params.f0 + self.eqns[0].params.beta*self.grid.Ymid)
        F = (self.eqns[0].params.f0 + 
              self.eqns[0].params.beta*(self.grid.Ymid - self.grid.Ymid.mean()))
        
        r, c = self.grid.hField.shape
        self.grid.hField += 1*np.random.randn(r, c)*(F/self.eqns[0].params.f0)
        # self.grid.hField += 1*np.random.uniform(size=(r,c))*(F/self.eqns[0].params.f0)
        # self.grid.hField += np.random.uniform(size=self.grid.hField.shape)

        # self.grid.hField += 1.0*np.random.randn(r,c)*(self.grid.dx/1.0e5)*(np.abs(F)/1e-16);