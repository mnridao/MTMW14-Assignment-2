"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from analyticalSolution import analyticalSolution
from equations import UVelocity, VVelocity, Eta
from grids import ArakawaCGrid
from model import Model
from solver import Solver

import wavesDemo
import plotters
import helpers

def runTaskC():
    """
    Runs Task C for Assignment 2. 
    
    Calculates the steady state solution for an ocean gyre according to 
    Mushgrave (1985) and plots the contours of the u- and v-velocities and the 
    height perturbation (eta).
    """
    # Compute the analytical solution with eta0 = 0.
    X = solver.model.grid.Xmid
    Y = solver.model.grid.Ymid
    params = solver.model.eqns[0].params
    uSol, vSol, etaSol = analyticalSolution(X, Y, xbounds[1], params, eta0=0.)
    
    # Plot the results.
    plotters.plotContourSubplot(uSol, vSol, etaSol, (X/1e3, Y/1e3))

def runTaskD1():
    """ 
    Runs the solver for 1 day (Task D1).
    """
    
    # Change the endtime to 1 day.
    endtime = 24*60**2
    nt = int(np.ceil(endtime/dt))
    
    # Create new solver so that other tasks aren't affected.
    solver = Solver(model, scheme, dt, nt)
    solver.run()
    
    # Plot results.
    plotters.plotSolutionSectionsFromGrid(solver.model.grid)
    
#### GLOBAL GRID VARIABLES ####
xbounds = [0, 1e6]
dx = 50e3
nx = int((xbounds[1] - xbounds[0])/dx)
grid = ArakawaCGrid(xbounds, nx, periodicX=False)

#### GLOBAL TIME STEPPING VARIABLES ####
dt = 0.85*helpers.calculateTimestepCFL(100, dx)
endtime = 50*24*60**2 
nt = int(np.ceil(endtime/dt))

#### GLOBAL MODEL AND SOLVER ####
model  = Model([Eta(), UVelocity(), VVelocity()], grid)
scheme = helpers.setScheme("forwardBackward")
solver = Solver(model, scheme, dt, nt)

if __name__ == "__main__":
        
    # Run Task C.    
    runTaskC()
    
    # Run Task D.1
    runTaskD1()
    
    #%% Task E
    solver.model.grid.resetFields()
    
    solver.addCustomEquations("energy", helpers.calculateEnergyModel)
    solver.run()
    energy = solver.getCustomData("energy").copy()
    
    # Energy at half the timestep.
    solver.model.grid.resetFields()
    solver.model.grid = ArakawaCGrid(xbounds, nx*2)
    
    dtHalf = 0.85*helpers.calculateTimestepCFL(100, solver.model.grid.dx*0.5)
    solver.setNewTimestep(dtHalf, endtime)
    solver.run()
    energyHalf = solver.getCustomData("energy")
    
    #%% Plotting
    # Calculate energy from analytical solution (eta0 = 0).
    uSS0, vSS0, etaSS0 = analyticalSolution(solver.model.grid.X, 
                                            solver.model.grid.Y,
                                            solver.model.grid.xbounds[1], 
                                            solver.model.eqns[0].params)
    energyAnalytical0 = helpers.calculateEnergy(uSS0, vSS0, etaSS0,
                                                solver.model.grid.dx, 
                                                solver.model.eqns[0].params)
    
    # Interpolate eta onto analytical grid.
    interpolator = RegularGridInterpolator((solver.model.grid.Ymid[:, 0], solver.model.grid.Xmid[0, :]), 
                                           solver.model.grid.hField, 
                                           bounds_error=False, fill_value=None)
    
    # Calculate energy from updated analytical solution (eta0 = numerical SS).
    eta0 = interpolator((solver.model.grid.Y, solver.model.grid.X))
    uSS, vSS, etaSS = analyticalSolution(solver.model.grid.X, 
                                          solver.model.grid.Y,
                                          solver.model.grid.xbounds[1], 
                                          solver.model.eqns[0].params,
                                          eta0)
    energyAnalytical = helpers.calculateEnergy(uSS, vSS, etaSS,
                                               solver.model.grid.dx, 
                                               solver.model.eqns[0].params)
        
    # Plot energy.
    time = np.arange(0, dt*(nt + 1), dt)/(24*60**2)
    plt.plot(figsize=(10, 10))
    
    # Energy from numerical solution.
    plt.plot(time, energy, linewidth=0.95)
    
    # Energy from analytical solution with eta0 = 0.
    plt.plot(time, energyAnalytical0*np.ones_like(time), 'k--', linewidth=0.95)
    
    # Energy from analytical solution with eta0 = steady state numerical solution.
    plt.plot(time, energyAnalytical*np.ones_like(time), 'k', linewidth=0.95)
    
    # Half grid spacing.
    timeHalf = np.arange(0, solver.dt*(solver.nt + 1), solver.dt)/(24*60**2)
    plt.plot(timeHalf, energyHalf, linewidth=0.95)
    
    plt.grid()
    plt.xlim([time.min(), time.max()])
    plt.xlabel("Time [days]", fontsize=10)
    plt.ylabel("Energy [J]", fontsize=10)
    
    plt.show()
    
    #%% Plot the difference in energy between steady state with grid spacing.
    endtime = 40*24*60**2 
    dx0 = 50e3
    dxScale = np.linspace(0.1, 1, 19)
    
    energies, energiesDiff = [], []
    for i, scale in enumerate(dxScale):
        solver.model.grid.resetFields()
        
        # Create grid with new resolution.
        nxi = int((xbounds[1] - xbounds[0])/(dx0*scale))
        solver.model.grid = ArakawaCGrid(xbounds, nxi)
        
        # Update timestep based on CFL.
        dti = 0.8*helpers.calculateTimestepCFL(100, dx0*scale)
        solver.setNewTimestep(dti, endtime)
        
        # Run the model and calculate energy.
        solver.run()
        
        # Use the finest resolution for the calculation of eta0.
        if i == 0:
            interpolator = RegularGridInterpolator((solver.model.grid.Ymid[:, 0], solver.model.grid.Xmid[0, :]), 
                                                   solver.model.grid.hField, 
                                                   bounds_error=False, fill_value=None)
        
        # Calculate steady state solution on new grid.
        eta0 = interpolator((solver.model.grid.Y, solver.model.grid.X))
        uSS, vSS, etaSS = analyticalSolution(solver.model.grid.X, 
                                              solver.model.grid.Y,
                                              solver.model.grid.xbounds[1], 
                                              solver.model.eqns[0].params,
                                              eta0)
        
        # Difference fields.
        uDiff = solver.model.grid.uField - 0.5*(uSS[1:, :] + uSS[:-1, :])
        vDiff = solver.model.grid.vField - 0.5*(vSS[:, 1:] + vSS[:, -1:])
        hDiff = solver.model.grid.hField - 0.25*(etaSS[:-1, :-1] + etaSS[1:, :-1] + etaSS[:-1, 1:] + etaSS[1:, 1:])
        
        energyDiff = helpers.calculateEnergy(uDiff, vDiff, hDiff,
                                             solver.model.grid.dx, 
                                             solver.model.eqns[0].params)
        
        energies.append(solver.getCustomData("energy"))
        
        # Energy difference at steady state.
        energiesDiff.append(energyDiff)
            
    # Differences in energy
    plt.figure(figsize=(10, 10))
    
    plt.plot(dx0*dxScale/1000, energiesDiff)
    
    plt.grid()
    plt.tick_params(labelsize=12)
    plt.xlabel("$\Delta$x [km]", fontsize=15)
    plt.ylabel("Energy difference [J]", fontsize=15)
    plt.show()
    
    
    #%%
    # Plot energy.
    time = np.arange(0, solver.dt*(solver.nt + 1), solver.dt)/(24*60**2)
    plt.plot(figsize=(10, 10))
    
    # Energy from numerical solution.
    plt.plot(time, energies[-1], linewidth=0.95)
    
    # # Energy from analytical solution with eta0 = 0.
    # plt.plot(time, energyAnalytical0*np.ones_like(time), 'k--', linewidth=0.95)
    
    # Energy from analytical solution with eta0 = steady state numerical solution.
    plt.plot(time, energyAnalytical*np.ones_like(time), 'k', linewidth=0.95)
    
    plt.grid()
    plt.xlim([time.min(), time.max()])
    plt.xlabel("Time [days]", fontsize=10)
    plt.ylabel("Energy [J]", fontsize=10)
    
    plt.show()
    
    #%%
    
    # # u vs x along the grid, closest to the southern edge of the basin.
    # plt.figure(figsize=(10, 10))
    # plt.plot(solver.model.grid.X[-1, :]/1000, solver.model.grid.uField[-1, :])
    # plt.grid()
    
    # plt.xlabel("X [km]", fontsize=20)
    # plt.ylabel("U [m/s]", fontsize=20)
    
    # plt.xlim([solver.model.grid.X[-1, :].min()/1000,
    #           solver.model.grid.X[-1, :].max()/1000])
    # plt.ylim([0, 1.05*solver.model.grid.uField[-1, :].max()])
    # plt.tick_params(labelsize=15)
    
    # # v vs y along the grid, closest to the western edge of the basin. 
    # plt.figure(figsize=(10, 10))
    # plt.plot(solver.model.grid.Y[:, 0]/1000, solver.model.grid.vField[:, 0])
    # plt.grid()
    
    # plt.xlabel("Y [km]", fontsize=20)
    # plt.ylabel("v [m/s]", fontsize=20)
    
    # plt.xlim([solver.model.grid.Y[:, 0].min()/1000,
    #           solver.model.grid.Y[:, 0].max()/1000])
    # plt.ylim([solver.model.grid.vField[:, 0].min(), 
    #           1.05*solver.model.grid.vField[:, 0].max()])
    # plt.tick_params(labelsize=15)
    
    # # eta vs x through the middle of the gyre.
    # plt.figure(figsize=(10, 10))
    
    # mid = int(0.5*solver.model.grid.Xmid.shape[0])
    # plt.plot(solver.model.grid.Xmid[mid, :]/1000, 
    #          solver.model.grid.hField[mid, :])
    # plt.grid()
    
    # plt.xlabel("X [km]", fontsize=20)
    # plt.ylabel("$\eta$ [m]", fontsize=20)
    
    # plt.xlim([solver.model.grid.Xmid[mid, :].min()/1000,
    #           solver.model.grid.Xmid[mid, :].max()/1000])
    # plt.ylim([solver.model.grid.hField[mid, :].min(), 
    #           1.05*solver.model.grid.hField[mid, :].max()])
    # plt.tick_params(labelsize=15)

    # # Height perturbation contour plot with streamlines.
    # plt.figure(figsize=(9, 12))
    # plt.streamplot(solver.model.grid.Xmid/1000, solver.model.grid.Ymid/1000, 
    #                solver.model.grid.uOnEtaField(), 
    #                solver.model.grid.vOnEtaField())
    # cont = plt.contourf(solver.model.grid.Xmid/1000, 
    #                     solver.model.grid.Ymid/1000, 
    #                     solver.model.grid.hField, levels=75)
    # plt.colorbar(cont, orientation='horizontal', pad=0.08)
    # plt.tick_params(labelsize=13)
    # plt.xlabel("X [km]", fontsize=20)
    # plt.ylabel("Y [km]", fontsize=20)
    
    # #%%
    # fig, axs = plt.subplots(2, 2, figsize=(20, 18))

    # # Plot u vs x along the grid, closest to the southern edge of the basin
    # axs[0, 0].plot(solver.model.grid.X[-1, :]/1000, solver.model.grid.uField[-1, :])
    # axs[0, 0].grid()
    # axs[0, 0].set_xlabel("X [km]", fontsize=20)
    # axs[0, 0].set_ylabel("U [m/s]", fontsize=20)
    # axs[0, 0].tick_params(labelsize=15)
    
    # # Plot v vs y along the grid, closest to the western edge of the basin
    # axs[0, 1].plot(solver.model.grid.Y[:, 0]/1000, solver.model.grid.vField[:, 0])
    # axs[0, 1].grid()
    # axs[0, 1].set_xlabel("Y [km]", fontsize=20)
    # axs[0, 1].set_ylabel("v [m/s]", fontsize=20)
    # axs[0, 1].tick_params(labelsize=15)
    
    # # Plot eta vs x through the middle of the gyre
    # mid = int(0.5 * solver.model.grid.Xmid.shape[0])
    # axs[1, 0].plot(solver.model.grid.Xmid[mid, :]/1000, solver.model.grid.hField[mid, :])
    # axs[1, 0].grid()
    # axs[1, 0].set_xlabel("X [km]", fontsize=20)
    # axs[1, 0].set_ylabel("$\eta$ [m]", fontsize=20)
    # axs[1, 0].tick_params(labelsize=15)
    
    # # Plot height perturbation contour plot with streamlines
    # cont = axs[1, 1].contourf(solver.model.grid.Xmid/1000, solver.model.grid.Ymid/1000, solver.model.grid.hField, levels=75)
    # axs[1, 1].streamplot(solver.model.grid.Xmid/1000, solver.model.grid.Ymid/1000, solver.model.grid.uOnEtaField(), solver.model.grid.vOnEtaField(), color='black')
    # plt.colorbar(cont, ax=axs[1, 1], orientation='horizontal', pad=0.08)
    # axs[1, 1].tick_params(labelsize=13)
    # axs[1, 1].set_xlabel("X [km]", fontsize=20)
    # axs[1, 1].set_ylabel("Y [km]", fontsize=20)
    
    # plt.tight_layout()
    # plt.show()
    
    #%%
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

# # Create a 4x2 grid of subplots with appropriate height ratios
# fig = plt.figure(figsize=(20, 10))
# gs = GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 1, 5])

# # Plot u vs x along the grid, closest to the southern edge of the basin
# ax1 = fig.add_subplot(gs[0, :2])
# ax1.plot(solver.model.grid.X[-1, :]/1000, solver.model.grid.uField[-1, :])
# ax1.grid()
# ax1.set_xlabel("X [km]", fontsize=20)
# ax1.set_ylabel("U [m/s]", fontsize=20)
# ax1.set_title("a)  ", loc="left", fontsize=20)
# ax1.tick_params(labelsize=15)

# # Plot v vs y along the grid, closest to the western edge of the basin
# ax2 = fig.add_subplot(gs[1, :2])
# ax2.plot(solver.model.grid.Y[:, 0]/1000, solver.model.grid.vField[:, 0])
# ax2.grid()
# ax2.set_xlabel("Y [km]", fontsize=20)
# ax2.set_ylabel("v [m/s]", fontsize=20)
# ax2.set_title("b) ", loc="left", fontsize=20)
# ax2.tick_params(labelsize=15)

# # Plot eta vs x through the middle of the gyre
# mid = int(0.5 * solver.model.grid.Xmid.shape[0])
# ax3 = fig.add_subplot(gs[2, :2])
# ax3.plot(solver.model.grid.Xmid[mid, :]/1000, solver.model.grid.hField[mid, :])
# ax3.grid()
# ax3.set_xlabel("X [km]", fontsize=20)
# ax3.set_ylabel("$\eta$ [m]", fontsize=20)
# ax3.set_title("c) ", loc="left", fontsize=20)
# ax3.tick_params(labelsize=15)

# # Plot height perturbation contour plot with streamlines
# ax4 = fig.add_subplot(gs[:, 2])
# cont = ax4.contourf(solver.model.grid.Xmid/1000, solver.model.grid.Ymid/1000, solver.model.grid.hField, levels=75)
# ax4.streamplot(solver.model.grid.Xmid/1000, solver.model.grid.Ymid/1000, solver.model.grid.uOnEtaField(), solver.model.grid.vOnEtaField(), color='black')
# cbar = plt.colorbar(cont, ax=ax4, orientation='vertical', pad=0.08)
# cbar.ax.tick_params(labelsize=13)
# cbar.set_label('$\eta$ [m]', fontsize=20)
# ax4.set_xlabel("X [km]", fontsize=20)
# ax4.set_ylabel("Y [km]", fontsize=20)
# ax4.set_title("d), ", loc="left", fontsize=20)

# plt.subplots_adjust(wspace=0.3)  # Adjust the horizontal space between subplots
# plt.tight_layout()
# plt.show()

# #%%

# plotters.plotSolutionSectionsFromGrid(solver.model.grid)
