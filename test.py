"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt

from equations import UVelocity, VVelocity, Eta
from grids import ArakawaCGrid

if __name__ == "__main__":
    
    # Grid spacing information.
    x0 = 0 
    xL = 1e6 
    dx = 50e3
    nx = int((xL - x0)/dx)
    
    # Initialise grid
    grid = ArakawaCGrid([x0, xL], nx)
    
    # Time stepping information.
    dt = 350
    endtime = 10*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    # Model
    etaEqn = Eta()
    uVelocityEqn = UVelocity()
    vVelocityEqn = VVelocity()
    #%%
    
    for t in range(nt):
        
        # Update height perturbation field.        
        grid.hField += dt*etaEqn(grid)
        
        # Forward backward euler.
        if t % 2 == 0:
            
            # Update the u-velocity field.
            grid.uField[:, 1:-1] += dt*uVelocityEqn(grid)
            
            # Update the v-velocity field.
            grid.vField[1:-1, :] += dt*vVelocityEqn(grid)
            
        else:
                        
            # Update the v-velocity field.
            grid.vField[1:-1, :] += dt*vVelocityEqn(grid)
            
            # Update the u-velocity field.            
            grid.uField[:, 1:-1] += dt*uVelocityEqn(grid)
        
    #%%
    
    # Plot to check
        
    # Plot contours
    fig, axs = plt.subplots(1, 3, figsize=(32, 13))

    cont1 = axs[0].imshow(grid.uField)
    plt.colorbar(cont1, location='bottom')
    axs[0].set_xlabel("X", fontsize=25)
    axs[0].set_ylabel("Y", fontsize=25)
    axs[0].set_title("u", fontsize=25)
        
    cont2 = axs[1].imshow(grid.vField)
    plt.colorbar(cont2, location='bottom')
    axs[1].set_xlabel("X", fontsize=25)
    axs[1].set_title("v", fontsize=25)

    cont3 = axs[2].imshow(grid.hField)
    plt.colorbar(cont3, location='bottom')
    # axs[2].contour(XS, YS, uSol, colors='black')
    axs[2].set_xlabel("X", fontsize=25)
    axs[2].set_title("$\eta$", fontsize=25)

    plt.show()