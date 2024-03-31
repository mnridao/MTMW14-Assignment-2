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
    sy = params.g*params.H * (dt/grid.dy)**2
        
    # Represent the terms right and left of the current point (ij)
    offDiagXTerms = [-sx]*(grid.nx - 1) + [0.]
    offDiagXTerms *= grid.nx
    
    # Represent the terms on rows above and below the current point (ij).
    offDiagYTerms = [-sy]*grid.ny*(grid.ny - 1)
        
    # Add off-diagonal elements to L.
    L = (np.diag(offDiagXTerms[:-1], k= 1) + 
         np.diag(offDiagXTerms[:-1], k=-1) + 
         np.diag(offDiagYTerms, k= grid.nx) + 
         np.diag(offDiagYTerms, k=-grid.nx))
    
    # Account for periodic boundary conditions.
    if grid.periodicX:
        periodicXTerms = [-sx] + [0.]*(grid.nx-1)
        periodicXTerms *= (grid.nx-1)
        periodicXTerms += [-sx]
        
        L += np.diag(periodicXTerms, k= grid.nx-1)
        L += np.diag(periodicXTerms, k=-grid.nx+1)
        
    # Add diagonal elements to L.
    L += np.diag((1 - np.sum(L, axis=1)))
    
    return L

if __name__ == "__main__":
    
    # Grid creation.
    xbounds = [0, 1e6]
    xL = xbounds[1]
    dx = 10e3
    nx = int((xbounds[1] - xbounds[0])/dx)
    grid = ArakawaCGrid(xbounds, nx, periodicX=True)
    
    eqns = [Eta(), UVelocity(), VVelocity()]
    
    dt = 300
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
        
        # Calculate A on u-grid.
        A = grid.uField.copy()
        
        # Account for different boundary conditions.
        if grid.periodicX:
            A += dt*(eqns[1].explicitTerms(grid))
        else:
            A[:, 1:-1] += dt*(eqns[1].explicitTerms(grid))
        
        # Calculate B on v-grid.
        B = grid.vField.copy()
        
        if grid.periodicY:
            B += dt*(eqns[2].explicitTerms(grid))
        else:
            B[1:-1, :] += dt*(eqns[2].explicitTerms(grid))
        
        # Calculate C on eta-grid.
        C = grid.hField.copy()
        
        # Calculate gradients of A and B (on eta grid).        
        dAdx = grid.forwardGradientFieldX(A)
        dBdy = grid.forwardGradientFieldY(B)
                
        # Calculate and flatten (row stack) explicit forcings array.
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