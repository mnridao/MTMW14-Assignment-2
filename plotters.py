"""
MTMW14 Assignment 2

Student ID: 31827379
"""
import matplotlib.pyplot as plt
    
def plotContourSubplot(u, v, eta, gridU, gridV=None, gridEta=None):
    """ 
    Contour plots of each of the fields stored in the grid.
    
    Inputs
    -------
    u        : np array
       Array containing the u-velocity field.
    v        : np array
       Array containing the v-velocity field.
    eta      : np array
       Array containing the height perturbation field (eta).
     gridU   : tuple of np arrays
        Grid information of u-velocity, gridU = (Xu, Yu)
     gridV   : tuple of np arrays
        Grid information of v-velocity, gridV = (Xv, Yv)
     gridEta : tuple of np arrays
        Grid information of eta, gridEta = (Xeta, Yeta)
    """
    
    # Set domain - all grids are the same if evaluated at the same point.
    gridV = gridV if gridV else gridU
    gridEta = gridEta if gridEta else gridU
    
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    
    # Subplot for u-velocity solution.
    cont1 = axs[0].contourf(*gridU, u, levels=75)
    plt.colorbar(cont1, ax=axs[0], orientation='horizontal')
    axs[0].set_xlabel("X [km]", fontsize=18)
    axs[0].set_ylabel("Y [km]", fontsize=18)
    axs[0].set_title("u [m/s]", fontsize=18)
    axs[0].tick_params(labelsize=13)
    
    # SUbplot for v-velocity solution.
    cont2 = axs[1].contourf(*gridV, v, levels=75)
    plt.colorbar(cont2, ax=axs[1], orientation='horizontal')
    axs[1].set_xlabel("X [km]", fontsize=18)
    axs[1].set_title("v [m/s]", fontsize=18)
    axs[1].tick_params(labelsize=13)
    
    # Subplot for eta solution.
    cont3 = axs[2].contourf(*gridEta, eta, levels=75)
    plt.colorbar(cont3, ax=axs[2], orientation='horizontal')
    axs[2].set_xlabel("X [km]", fontsize=18)
    axs[2].set_title("$\eta$ [m]", fontsize=18)
    axs[2].tick_params(labelsize=13)
    
    plt.show()
    
def plotSolutionSectionsFromGrid(grid):
    """ 
    Plots different sections of the gyre for the current state of the grid.
    Left plot: 
        - u-velocity along the southern edge of the basin
        - v-velocity along the western edge of the basin 
        - eta vs X through the middle of the gyre
    Right plot:
        - contour plot of eta on the domain
        - streamplot of the velocity field overlaying the contour plot
    
    Inputs
    -------
    grid : ArakawaCGrid object
           Object containing the domain and state information for the problem.
    
    """    
    fontsize=10
    fontsizeTicks = 6
    
    fig = plt.figure(figsize=(10, 3.5))
    
    # Create a single subplot for u, v and eta.
    ax1 = fig.add_subplot(1, 2, 1)  
    
    # Plot u vs x along the southern edge of the basin.
    ax1.plot(grid.X[-1, :]/1000, grid.uField[-1, :], linewidth=0.75,
             label='u-velocity: southern edge')
    
    # Plot v vs y along the western edge of the basin.
    ax1.plot(grid.Y[:, 0]/1000, grid.vField[:, 0], linewidth=0.75, 
             label='v-velocity: western edge')
    
    # Plot eta vs x through the middle of the gyre.
    mid = int(0.5 * grid.Xmid.shape[0])
    ax1.plot(grid.Xmid[mid, :]/1000, grid.hField[mid, :], linewidth=0.75,
             label='$\eta$: horizontal midpoint')
    
    # Set x limits (assumes the X and Y domains are the same length).
    ax1.set_xlim([grid.xbounds[0]/1000, grid.xbounds[1]/1000])
    
    # Add grid and legend
    ax1.grid()
    ax1.legend(fontsize=fontsizeTicks)
    ax1.set_xlabel("X [km], Y[km]", fontsize=fontsize)
    ax1.set_ylabel("u [m/s], v [m/s], $\eta$ [m]", fontsize=fontsize)
    ax1.tick_params(labelsize=fontsizeTicks)
    
    # Plot height perturbation contour plot with streamlines
    ax2 = fig.add_subplot(1, 2, 2)
    cont = ax2.contourf(grid.Xmid/1000, grid.Ymid/1000, grid.hField, levels=75)
    ax2.streamplot(grid.Xmid/1000, grid.Ymid/1000, grid.uOnEtaField(), 
                   grid.vOnEtaField(), color='black', density=[0.7, 1], linewidth=0.5)
    cbar = plt.colorbar(cont, ax=ax2, orientation='vertical')
    
    cbar.ax.tick_params(labelsize=fontsizeTicks)
    ax2.set_xlabel("X [km]", fontsize=fontsize)
    ax2.set_ylabel("Y [km]", fontsize=fontsize)
    ax2.set_title('$\eta$ [m]', fontsize=fontsize)
    
    plt.tight_layout()
    plt.show()