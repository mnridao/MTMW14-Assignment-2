"""
MTMW14 Assignment 2

Student ID: 31827379
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analyticalSolution import analyticalSolution
import helpers
    
def plotContourSubplot(u, v, eta, gridU, gridV=None, gridEta=None,
                       uComp=None, vComp=None, etaComp=None, levels=75):
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
        
    # Fontsizes.
    fontsize = 10
    tickSize = 6
    figsize = (11, 4)
    
    # Set domain - all grids are the same if evaluated at the same point.
    gridV = gridV if gridV else gridU
    gridEta = gridEta if gridEta else gridU
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Subplot for u-velocity solution.
    uX, uY = gridU[0]/1e3, gridU[1]/1e3
    
    if type(uComp) != type(None):
        axs[0].contour(uX, uY, uComp, colors='black', alpha=0.5)
    cont1 = axs[0].contourf(uX, uY, u, levels=levels)
    col1 = plt.colorbar(cont1, ax=axs[0], orientation='horizontal')
        
    col1.ax.tick_params(labelsize=tickSize)
    axs[0].set_xlabel("X [km]", fontsize=fontsize)
    axs[0].set_ylabel("Y [km]", fontsize=fontsize)
    axs[0].set_title("u [m/s]", fontsize=fontsize)
    axs[0].tick_params(labelsize=tickSize)
    
    # SUbplot for v-velocity solution.
    vX, vY = gridV[0]/1e3, gridV[1]/1e3
    cont2 = axs[1].contourf(vX, vY, v, levels=levels)
    col2 = plt.colorbar(cont2, ax=axs[1], orientation='horizontal')
    col2.ax.tick_params(labelsize=tickSize)
    
    if type(vComp) != type(None):
        axs[1].contour(vX, vY, vComp, colors='black', alpha=0.5)
    
    axs[1].set_xlabel("X [km]", fontsize=fontsize)
    axs[1].set_title("v [m/s]", fontsize=fontsize)
    axs[1].tick_params(labelsize=tickSize)
    
    # Subplot for eta solution.
    etaX, etaY = gridEta[0]/1e3, gridEta[1]/1e3
    cont3 = axs[2].contourf(etaX, etaY, eta, levels=levels)
    col3 = plt.colorbar(cont3, ax=axs[2], orientation='horizontal')
    
    if type(etaComp) != type(None):
        axs[2].contour(etaX, etaY, etaComp, colors='black', alpha=0.5)
    
    col3.ax.tick_params(labelsize=tickSize)
    axs[2].set_xlabel("X [km]", fontsize=fontsize)
    axs[2].set_title("$\eta$ [m]", fontsize=fontsize)
    axs[2].tick_params(labelsize=tickSize)
    
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
    ax1.plot(grid.X[0, :]/1000, grid.uField[0, :], linewidth=0.75,
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
    ax2.set_xlim([grid.xbounds[0]/1e3, grid.xbounds[1]/1e3])
    ax2.set_ylim([grid.ybounds[0]/1e3, grid.ybounds[1]/1e3])
    ax2.set_title('$\eta$ [m]', fontsize=fontsize)
    
    plt.tight_layout()
    plt.show()
    
def plotSteadyStateWithAnalytical(model):
    """ 
    Plots different sections of the gyre for the current state of the grid and
    compares with the analytical solution (interpolated on each respective
    field). For this project, I have only used this function for evenly sized 
    eta fields - would affect how the midpoint eta is calculated.
    
    3 left plots: 
        - u-velocity along the southern edge of the basin
        - v-velocity along the western edge of the basin 
        - eta vs X through the middle of the gyre
    Right plot:
        - contour plot of eta on the domain
        - streamplot of the velocity field overlaying the contour plot
        
    Inputs
    ------
    model : Model object
    """
    
    # Fontsizes.
    fontsize = 8
    tickSize = 7
    figsize = (8.7, 4.1)
    
    # Calculate analytical solution.
    eta0 = helpers.calculateEta0(model.grid.hField)
    uSS, vSS, etaSS = analyticalSolution(model.grid, model.eqns[0].params, eta0)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1.5, 1, 5])
    
    # Plot u vs x along the southern edge of the basin.
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(model.grid.X[0, :]/1000, uSS[0, :], 'k--', 
             linewidth=0.75)
    ax1.plot(model.grid.X[0, :]/1000, model.grid.uField[0, :], linewidth=0.75)
    ax1.grid()
    ax1.set_xlabel("X [km]", fontsize=fontsize)
    ax1.set_ylabel("U [m/s]", fontsize=fontsize)
    # ax1.set_title("a)  ", loc="left", fontsize=fontsize)
    ax1.tick_params(labelsize=tickSize)
    
    # Plot v vs y along the western edge of the basin.
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(model.grid.Y[:, 0]/1000, vSS[:, 0], 'k--', linewidth=0.75)
    ax2.plot(model.grid.Y[:, 0]/1000, model.grid.vField[:, 0], linewidth=0.75)
    ax2.grid()
    ax2.set_xlabel("Y [km]", fontsize=fontsize)
    ax2.set_ylabel("v [m/s]", fontsize=fontsize)
    # ax2.set_title("b) ", loc="left", fontsize=fontsize)
    ax2.tick_params(labelsize=tickSize)
    
    # Plot eta vs x through the middle of the gyre.
    mid = int(0.5 * model.grid.Xmid.shape[0])
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(model.grid.Xmid[mid, :]/1000, etaSS[mid, :], 'k--', 
             label="Analytical", linewidth=0.75)
    ax3.plot(model.grid.Xmid[mid, :]/1000, 
             0.5*(model.grid.hField[mid-1, :] + model.grid.hField[mid, :])
             ,label="numerical", linewidth=0.75)
    ax3.grid()
    plt.legend(fontsize=tickSize)
    ax3.set_xlabel("X [km]", fontsize=fontsize)
    ax3.set_ylabel("$\eta$ [m]", fontsize=fontsize)
    # ax3.set_title("c) ", loc="left", fontsize=fontsize)
    ax3.tick_params(labelsize=tickSize)
    
    # Plot height perturbation contour plot with streamlines.
    ax4 = fig.add_subplot(gs[:, 2])
    cont = ax4.contourf(model.grid.Xmid/1000, model.grid.Ymid/1000, 
                        model.grid.hField, levels=75)
    ax4.streamplot(model.grid.Xmid/1000, model.grid.Ymid/1000, 
                   model.grid.uOnEtaField(), model.grid.vOnEtaField(), color='black',
                   density=[0.7, 1], linewidth=0.5)
    cbar = plt.colorbar(cont, ax=ax4, orientation='vertical', pad=0.08)
    
    cbar.ax.tick_params(labelsize=tickSize)
    cbar.set_label('$\eta$ [m]', fontsize=fontsize)
    ax4.set_xlim([model.grid.xbounds[0]/1e3, model.grid.xbounds[1]/1e3])
    ax4.set_ylim([model.grid.ybounds[0]/1e3, model.grid.ybounds[1]/1e3])
    ax4.set_xlabel("X [km]", fontsize=fontsize)
    ax4.set_ylabel("Y [km]", fontsize=fontsize)
    # ax4.set_title("d), ", loc="left", fontsize=fontsize)
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.tight_layout()
    plt.show()
    
def plotEnergiesTaskE(dxs, energies, energyDiffs, timeTaken, model, endtime):
    """ 
    Plots the time evolution of energy for different values of dx, and also 
    plots the energy difference between the numerical and anlytical solutions
    for different values of dx, as well as the CPU time taken to run each 
    model.
    
    Inputs
    ------
    dxs         : list of floats
       List of the different dx values that are being considered.
    energies    : list of numpy arrays
       List of the time evolution of energy for each value of dx.
    energyDiffs : List of 
    timeTaken   :
    model       :
    endtime     :
    """
    
    # Calculate energy of analytical solution for eta0 = 0.
    uSS0, vSS0, etaSS0 = analyticalSolution(model.grid, model.eqns[0].params)
    energySS0 = helpers.calculateEnergy(uSS0, vSS0, etaSS0, model.grid.dx, 
                                        model.eqns[0].params)
    
    # Calculate energy of analytical solution for eta0 = most recent model.
    eta0 = helpers.calculateEta0(model.grid.hField)
    uSS, vSS, etaSS = analyticalSolution(model.grid, model.eqns[0].params, eta0)

    energySS = helpers.calculateEnergy(uSS, vSS, etaSS, model.grid.dx, 
                                       model.eqns[0].params)
    
    # Figure properties - easy to change for jupyter notebook.
    figsize=(8.5, 4)
    fontsize=8
    tickSize=7
    markerSize=0.01
    
    # Plot time evolution of energy vs most recent analytical solution.
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # dxsSliced = dxs[::2]
    # energySliced = energies[::2]
    dxsSliced = dxs
    energySliced = energies
    for i, e in enumerate(energySliced):
        
        time = np.linspace(0, endtime, e.shape[0])/(24*60**2)
        axs[0].plot(time, e, label=f"$\Delta$x={dxsSliced[i]/1000:.0f}km",
                    linewidth=0.75)
        
        # # Plot analytical energies.
        # axs[0].plot(time, energySS0*np.ones_like(time), 'k--', linewidth=0.55)
        # axs[0].plot(time, energySS*np.ones_like(time), 'k', linewidth=0.55)
        
    axs[0].set_xlabel("Time [days]", fontsize=fontsize)
    axs[0].set_ylabel("Energy [J]", fontsize=fontsize)
    axs[0].legend(fontsize=tickSize)
    axs[0].set_xlim([time.min(), time.max()])
    axs[0].tick_params(labelsize=tickSize)
    axs[0].grid()

    # Plot the energy difference vs dx on the right subplot.
    axs[1].plot([dxi/1000 for dxi in dxs], energyDiffs, 'o-', 
                label="Energy difference", linewidth=0.75, markeredgewidth=markerSize)
    axs[1].grid(which="both")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("$\Delta$x [km]", fontsize=fontsize)
    axs[1].set_ylabel("Energy difference [J]", fontsize=fontsize)
    axs[1].tick_params(labelsize=tickSize)
    axs[1].set_xlim([0, dxs[0]/1e3])

    # Create a second axis sharing the same x-axis
    ax2 = axs[1].twinx()
    ax2.set_yscale("log")
    ax2.plot([dxi/1000 for dxi in dxs], timeTaken, 'ro-', 
             label='CPU time', linewidth=0.75, markeredgewidth=markerSize)
    ax2.set_ylabel('CPU time [s]', fontsize=fontsize)
    ax2.tick_params(labelsize=tickSize)

    # Combine legends from both axes
    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=tickSize)

    plt.show()
    
def plotContoursSchemes(hFields, uFields, vFields, schemes, grid,
                        hConts=None, uConts=None, vConts=None):
    """ 
    """

    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1]*3)
        
    levels = 25
    fontsize=15
    ticksize=12
    
    for i, s in enumerate(schemes):
        
        # Find minimum and maximum values for scale.
        hMin, hMax = hFields.min(), hFields.max()
        uMin, uMax = uFields.min(), uFields.max()
        vMin, vMax = vFields.min(), vFields.max()
        
        ## Plot eta field.
        ax1 = fig.add_subplot(gs[0, i])
        X1, Y1 = grid.etaGrid()
        cont1 = ax1.contourf(X1/1e3, Y1/1e3, hFields[i, ...],
                           levels=np.linspace(hMin, hMax, levels))
        if type(hConts) != type(None):
            ax1.contour(X1/1e3, Y1/1e3, hConts[i, ...], colors='black', alpha=0.5)
        
        ax1.tick_params(labelsize=ticksize)
        ax1.set_xticks([])
        ax1.set_title(s, fontsize=fontsize, loc="left")
        
        ## Plot u-velocity field.
        ax2 = fig.add_subplot(gs[1, i])
        X2, Y2 = grid.uGrid()
        cont2 = ax2.contourf(X2/1e3, Y2/1e3, uFields[i, ...],
                           levels=np.linspace(uMin, uMax, levels))
        if type(uConts) != type(None):
            ax2.contour(X2/1e3, Y2/1e3, uConts[i, ...], colors='black', alpha=0.5)
        ax2.tick_params(labelsize=ticksize)
        ax2.set_xticks([])
                
        ## Plot v-velocity field.
        ax3 = fig.add_subplot(gs[2, i])
        X3, Y3 = grid.vGrid()
        cont3 = ax3.contourf(X3/1e3, Y3/1e3, vFields[i, ...], 
                           levels=np.linspace(vMin, vMax, levels))
        if type(vConts) != type(None):
            ax3.contour(X3/1e3, Y3/1e3, vConts[i, ...], colors='black', alpha=0.5)
        ax3.tick_params(labelsize=ticksize)
        ax3.set_xlabel("X [km]", fontsize=fontsize)
        
        if i != 0:
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax3.set_yticks([])
        else:
            ax1.set_ylabel("Y [km]", fontsize=fontsize)
            ax2.set_ylabel("Y [km]", fontsize=fontsize)
            ax3.set_ylabel("Y [km]", fontsize=fontsize)
            
        if i == len(schemes)-1:
            
            # Add the eta colorbar.
            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(cont1, cax=cax1, orientation='vertical', pad=0.08)
            cbar1.ax.tick_params(labelsize=ticksize)
            cbar1.set_label('$\eta$ [m]', fontsize=fontsize)
            
            # Add the u-velocity colorbar.
            divider = make_axes_locatable(ax2)
            cax2 = divider.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(cont2, cax=cax2, orientation='vertical', pad=0.08)
            cbar2.ax.tick_params(labelsize=ticksize)
            cbar2.set_label('$u$ [m/s]', fontsize=fontsize)
            
            # Add the v-velocity colorbar.
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="5%", pad=0.05)
            cbar3 = plt.colorbar(cont3, cax=cax3, orientation='vertical', pad=0.08)
            cbar3.ax.tick_params(labelsize=ticksize)
            cbar3.set_label('$v$ [m/s]', fontsize=fontsize)
            
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, 
                        hspace=0.05, wspace=0.05)
    plt.show()
    
def plotAllSchemeSections(grid, hFields, uFields, vFields, uSS, vSS, etaSS, schemes):
    """ 
    """
    
    # Set parameters here (need to check what looks better).
    fontsize = 10
    ticksize = 6
    figsize = (11, 3)
            
    # Plot the gyre sections.
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Plot analytical.
    axs[0].plot(grid.X[0, :]/1e3, uSS[0, :], 'k--', linewidth=0.75, 
                label='Analytical')
    axs[1].plot(grid.Y[:, 0]/1e3, vSS[:, 0], 'k--', linewidth=0.75,
                label='Analytical')
    
    mid = int(0.5 * grid.Xmid.shape[0])
    axs[2].plot(grid.Xmid[mid, :]/1e3, etaSS[mid, :], 'k--', linewidth=0.75, 
                label='Analytical')
    
    for i, s in enumerate(schemes):
                        
        # Plot u vs x along the southern edge of the basin.
        axs[0].plot(grid.X[0, :]/1e3, uFields[i, 0, :], linewidth=0.75,
                    label=s)
        
        # Plot v vs y along the western edge of the basin.
        axs[1].plot(grid.Y[:, 0]/1e3, vFields[i, :, 0], linewidth=0.75, 
                    label=s)
        
        # Plot eta vs x through the middle of the gyre.
        axs[2].plot(grid.Xmid[mid, :]/1e3, hFields[i, mid, :], linewidth=0.75,
                    label=s)
    
    axs[0].set_xlabel("X [km]", fontsize=fontsize)
    axs[0].set_ylabel("u [m/s]", fontsize=fontsize)
    axs[0].set_title("$u$ on southern boundary [m/s]", fontsize=fontsize, loc="left")
    axs[0].set_xlim([grid.xbounds[0]/1e3, grid.xbounds[1]/1e3])
    axs[0].grid()
    axs[0].tick_params(labelsize=ticksize)
    
    axs[1].set_xlabel("Y [km]", fontsize=fontsize)
    axs[1].set_ylabel("v [m/s]", fontsize=fontsize)
    axs[1].set_title("$v$ on western boundary [m/s]", fontsize=fontsize, loc="left")
    axs[1].set_xlim([grid.xbounds[0]/1e3, grid.xbounds[1]/1e3])
    axs[1].grid()
    axs[1].tick_params(labelsize=ticksize)
    
    axs[2].set_xlabel("X [km]", fontsize=fontsize)
    axs[2].set_ylabel("$\eta$ [m/s]", fontsize=fontsize)
    axs[2].set_title("$\eta$ through gyre midpoint [m/s]", fontsize=fontsize, loc="left")
    axs[2].set_xlim([grid.xbounds[0]/1e3, grid.xbounds[1]/1e3])
    axs[2].grid()
    axs[2].legend(fontsize=ticksize)
    axs[2].tick_params(labelsize=ticksize)
    
    plt.tight_layout()
    plt.show()

def plotHeightContoursRow(hFields, uFields, vFields, schemes, grid, params):
    """ 
    """
    
    # Set parameters
    levels = 50
    fontsize = 15
    ticksize = 12
    figsize = (15, 5)
    
    # Find minimum and maximum values of eta for colorbar
    hMin, hMax = hFields.min(), hFields.max()
    
    # Calculate analytical solution
    eta0 = helpers.calculateEta0(hFields[0, ...])
    uSS, vSS, etaSS = analyticalSolution(grid, params, eta0)
    
    # Plot the height perturbation contours with streamlines
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Plot height perturbation contour plot with streamlines in each subplot
    for i, ax in enumerate(axs.flatten()):
        
        # Get the fields for the ith time scheme
        hField = hFields[i, ...]
        uField = uFields[i, ...]
        vField = vFields[i, ...]
        
        # Interpolate the velocity fields onto the eta field
        uOnEta = 0.5 * (uField[:, :-1] + uField[:, 1:])
        vOnEta = 0.5 * (vField[:-1, :] + vField[1:, :])
        
        cont = ax.contourf(grid.Xmid/1e3, grid.Ymid/1e3, hField,
                            levels=np.linspace(hMin, hMax, levels))
        
        ax.streamplot(grid.Xmid/1000, grid.Ymid/1000, uOnEta, vOnEta, 
                      color='black', linewidth=0.5, arrowsize=0.8, density=1.1)
        
        ax.set_title(schemes[i], fontsize=fontsize)
        ax.set_xlim([grid.xbounds[0]/1e3, grid.xbounds[1]/1e3])
        ax.set_ylim([grid.ybounds[0]/1e3, grid.ybounds[1]/1e3])
        
        ax.set_xlabel("X [km]", fontsize=fontsize)
        
        if i // 1 == 0:
            ax.set_ylabel("Y [km]", fontsize=fontsize)
        else:
            ax.set_yticks([])
    
    # Add a colorbar to the right of the last plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(cont, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('$\eta$ [m]', fontsize=fontsize)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    plt.show()

def plotTimeStepsTaskG(dts, energyDiffs, timeTaken, schemes):
    """ 
    """
    # Plot the things.
    figsize = (10, 4)
    fontsize = 15
    ticksize = 12
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Define the length of a day in seconds for scaling.
    day = 24*60**2
    
    # First subplot: Energy differences vs dts
    for i, si in enumerate(schemes):
        if si == "forwardBackward" or si == "semiLagrangian":
            dtsi = dts[:1]
            markersize=5
        elif si == "rk4":
            dtsi = dts[:2]
            markersize=6
        else:
            dtsi = dts
            markersize=5
        
        axs[0].plot(np.array(dtsi)/day, energyDiffs[i], '-o', label=si, 
                    markersize=markersize, linewidth=0.75)
        axs[1].plot(np.array(dtsi)/day, timeTaken[i], '-o', label=si,
                    linewidth=0.75)
    
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].grid(which="both")
    axs[0].set_xlabel("$\Delta$t [days]", fontsize=fontsize)
    axs[0].set_ylabel("Energy difference [J]", fontsize=fontsize)
    axs[0].set_title("a)", loc="left")
    axs[0].tick_params(labelsize=ticksize)
    
    axs[1].set_xscale("log")
    # axs[1].set_yscale("log")
    axs[1].grid(which="both")
    axs[1].set_xlabel("$\Delta$t [days]", fontsize=fontsize)
    axs[1].set_ylabel("CPU time [s]", fontsize=fontsize)
    axs[1].set_title("b)", loc="left")
    axs[1].tick_params(labelsize=ticksize)
    axs[1].legend(fontsize=ticksize)
    
    # Add labels to first two dots (less than day so confusing with log).
    axs[1].text((dts[0]+40)/day, timeTaken[2][0]+4, f"$\Delta$t={dts[0]:.0f}s (CFL)",
                    fontsize=0.9*ticksize, rotation=90, ha='right', va='bottom')
    axs[1].text((dts[1] + 75)/day, timeTaken[2][0]+2, f"$\Delta$t={dts[1]:.0f}s",
                    fontsize=0.9*ticksize, rotation=90, ha='right', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()