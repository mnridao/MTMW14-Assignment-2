"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

from grids import ArakawaCGrid
from model import Model
from equations import Parameters, UVelocity, VVelocity, Eta

from plotters import plotContourSubplot
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve 

def createImplicitCoefficientsMatrix(grid, params, dt):
    """ 
    """
    
    # Create terms for L matrix diagonal.
    sx = params.g*params.H * (dt/grid.dx)**2
    sy = params.g*params.H * (dt/grid.dx)**2
    
    # Shape of the L matrix.
    numCols = grid.hField.shape[1]
    numRows = grid.hField.shape[0]
    
    # Represent the terms right and left of the current point (ij)
    offDiagXTerms = [-sx]*(numCols - 1) + [0]
    offDiagXTerms *= numRows
    
    # Represent the terms on rows above and below the current point (ij).
    offDiagYTerms = [-sy]*numRows*(numRows - 1)
        
    # Full L matrix.
    L = (np.diag(offDiagXTerms[:-1], k=1) + 
         np.diag(offDiagXTerms[:-1], k=-1) + 
         np.diag(offDiagYTerms, k=numCols) + 
         np.diag(offDiagYTerms, k=-numCols))
    L += np.diag((1 - np.sum(L, axis=1)))
    
    return L

if __name__ == "__main__":
    
    # Grid creation.
    xbounds = [0, 1e6]
    xL = xbounds[1]
    dx = 10e3
    nx = int((xbounds[1] - xbounds[0])/dx)
    # nx = 254
    grid = ArakawaCGrid(xbounds, nx, periodicX=False)
    
    dt = 69
    endtime = 30*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    # Get the default parameters.
    params = Parameters()
        
    # Plot to see.
    plotContourSubplot(grid)
    
    #### L Matrix ####
    L = createImplicitCoefficientsMatrix(grid, params, dt)
    
    #%% Time loop.
    
    # nt = 100
    hFields = []
    for t in range(nt):
        
        ### CALCULATE A ON U GRID ####
        A = grid.uField.copy()           # Initially set A as current u.
        
        # Get the internal u points from grid.
        u = grid.fields["uVelocity"]
        
        # Coriolis parameter (at half grid points - assumes c grid).
        f = (params.f0 + params.beta*grid.Ymid)[..., :u.shape[1]]    
        
        # Wind forcing in x-direction.
        tauX = params.tauX(grid.Ymid, grid.xbounds[1])[..., :u.shape[1]]
        
        # Set the internal A points as forcings without gravity (boundaries are 0).
        A[:, 1:-1] = (f*grid.vOnUField() - params.gamma*u + tauX/(params.rho*params.H))
        
        ### CALCULATE B ON V GRID ####
        B = grid.vField.copy()            # Initially set B as current v.
        
        # Get the internal v points from grid.
        v = grid.fields["vVelocity"]
        Y = grid.Y[1:-1, :-1]
        
        # Coriolis parameter.
        f = params.f0 + params.beta*Y
        
        # Wind forcing in y-direction.        
        tauY = params.tauY(Y)
        
        # Set the internal B points as forcings without gravity (boundaries are 0).
        B[1:-1, :] = (-f*grid.uOnVField() - params.gamma*v + tauY/(params.rho*params.H))
        
        ### CALCULATE C ON ETA GRID ####
        C = grid.hField.copy()
        
        # Calculate gradients of A and B (on eta grid).
        dAdx = (A[:, 1:] - A[:, :-1]) / grid.dx
        dBdy = (B[1:, :] - B[:-1, :]) / grid.dy
                
        # Calculate and flatten explicit forcings array.
        F = (C - dt*params.H*(dAdx + dBdy)).flatten()
        
        # Update eta.
        # grid.hField = np.matmul(np.linalg.inv(L), F).reshape(grid.hField.shape)
        grid.hField = spsolve(L, F).reshape(grid.hField.shape)
        
        # Update velocity fields.
        grid.uField[:, 1:-1] = A[:, 1:-1] - dt*params.g * grid.detadxField()
        grid.vField[1:-1, :] = B[1:-1, :] - dt*params.g * grid.detadyField()
            
        # Update the viewers.
        grid.fields["eta"] = grid.hField 
        grid.fields["uVelocity"] = grid.uField[:, 1:-1]
        grid.fields["vVelocity"] = grid.vField[1:-1, :]
        
        # Plot to see.
        plotContourSubplot(grid)
                
        # Store h grid.
        hFields.append(grid.hField)