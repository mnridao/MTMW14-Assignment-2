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
    sx = params.g*params.H * 0.25*(dt/grid.dx)**2
    sy = params.g*params.H * 0.25*(dt/grid.dx)**2
        
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
    # nx = 254
    grid = ArakawaCGrid(xbounds, nx, periodicX=False)
    
    dt = 400
    # dt = 69
    
    # Get the default parameters.
    params = Parameters()
    
    # Add blob initial condition to eta field.
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    model.setBlobInitialCondition(xL*np.array([0.1, 0.5]), 
                                  ((2*dx)**2*np.array([2, 2])**2), 1*dx)
            
    # Just incase not updating.
    grid = model.grid
    
    # Plot to see.
    plotContourSubplot(grid)
    
    # Calculate the L grid (coefficient matrix of implicit terms).    
    L = createImplicitCoefficientsMatrix(grid, params, dt)
    Linv = np.linalg.inv(L)
    
    #%% Time loop
    
    nt = 100
    hFields = []
    for t in range(nt):
        
        # Calculate A (on u grid), B (on v grid) and C (on eta grid) matrices.
        A = grid.uField.copy()
        B = grid.vField.copy()
        C = grid.hField.copy()
        
        # Calculate gradients of A and B (on eta grid).
        dAdx = grid.forwardGradientFieldX(A)
        dBdy = grid.forwardGradientFieldY(B)
                
        # Calculate and flatten explicit forcings array.
        F = (C - dt*params.H*(dAdx + dBdy)).flatten()
        
        # Update eta.
        grid.hField = np.matmul(Linv, F).reshape(grid.hField.shape)
        
        # Update velocity fields.
        if grid.periodicX:
            grid.uField = A - dt*params.g * grid.detadxField()
        else:
            grid.uField[:, 1:-1] = A[:, 1:-1] - dt*params.g * grid.detadxField()
            
        grid.vField[1:-1, :] = B[1:-1, :] - dt*params.g * grid.detadyField()
            
        # Update the viewers (dont worry about this - just so grid is properly updated).
        grid.fields["eta"] = grid.hField 
        
        if grid.periodicX:
            grid.fields["uVelocity"] = grid.uField 
        else:
            grid.fields["uVelocity"] = grid.uField[:, 1:-1]
        grid.fields["vVelocity"] = grid.vField[1:-1, :]
        
        # Plot to see.
        plotContourSubplot(grid)
                
        # # Store h grid.
        # hFields.append(grid.hField)
    
#%%
    minH = min(np.min(state) for state in hFields)
    maxH = max(np.max(state) for state in hFields)
    
    for state in hFields:
        
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        
        # norm = colors.Normalize(vmin=minH, vmax=maxH)
        
        # Plot the surface
        surf = ax.plot_surface(grid.Xmid, 
                               grid.Ymid, 
                               state, 
                                cmap='viridis', 
                               rstride=5, cstride=5, antialiased=True,
                                # vmin=minH, vmax=maxH
                               )
        
        ax.set_zlim(minH, maxH)
        
        # Customize the plot
        ax.set_xlabel('X [km]', fontsize=25)
        ax.set_ylabel('Y [km]', fontsize=25)
    
        plt.show()
    
        #%%
    from matplotlib.animation import FuncAnimation
    
    # minH = min(np.min(state[2]) for state in hFields)
    minH = -2865.17755172608
    # maxH = max(np.max(state[2]) for state in hFields)
    maxH = 10000.0
    
    # Set up the figure and axis
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
    
        state = states[frame]
    
        # Plot the surface
        surf = ax.plot_surface(grid.Xmid,
                                grid.Ymid,
                                state,
                                cmap='viridis',
                                rstride=5, cstride=5, antialiased=True,
                                vmin=minH, vmax=maxH
                                )
    
        ax.set_zlim(minH, maxH)
        ax.set_xlabel('X [km]', fontsize=25)
        ax.set_ylabel('Y [km]', fontsize=25)
        ax.set_title(f'Frame {frame}', fontsize=25)
    
        return surf,
    
    # The total number of frames is the length of the solver history
    states = hFields
    total_frames = len(states)
        
    # Use FuncAnimation to create the animation
    animation = FuncAnimation(fig, update, frames=total_frames
                               , interval=200
                              )
    
    plt.show()
    
    # Save the animation as a GIF
    from matplotlib.animation import PillowWriter
    
    writer = PillowWriter(fps=30)
    animation.save("gravityWaveSemiImplicit.gif", writer=writer)
    
    # Close the figure to avoid displaying it twice
    plt.close(fig)