"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt
import time

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
    params = solver.model.eqns[0].params
    uSol, vSol, etaSol = analyticalSolution(solver.model.grid, params)
    
    # Plot the results.
    grid = solver.model.grid
    plotters.plotContourSubplot(uSol, vSol, etaSol, 
                                grid.uGrid(), grid.vGrid(), grid.etaGrid())

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

def runTaskD2():
    """ 
    Runs the solver until steady state, calculated as 40 days in Task E 
    (Task D2).
    """
    
    # Run the solver using global variables.
    solver.model.grid.resetFields()
    solver.run()
    
    # Plot the steady state and compare with the analytical solution.
    plotters.plotSteadyStateWithAnalytical(solver.model)
                
def runTaskD3():
    """
    Without resetting the solver, calculates the energy difference between the 
    analytical and numerical solution (Task D3).
    """
    
    # Run the solver.
    solver.model.grid.resetFields()
    solver.run()
    
    # Calculate the analytical solution.
    eta0 = helpers.calculateEta0(model.grid.hField)
    uSS, vSS, etaSS = analyticalSolution(model.grid, model.eqns[0].params, eta0)
    
    # Calculate the difference fields.    
    uDiff = model.grid.uField - uSS
    vDiff = model.grid.vField - vSS
    hDiff = model.grid.hField - etaSS
    
    uDiffNormalised = np.abs(uDiff) / np.max(np.abs(uDiff))
    vDiffNormalised = np.abs(vDiff) / np.max(np.abs(vDiff))
    hDiffNormalised = np.abs(hDiff) / np.max(np.abs(hDiff))
        
    # Plot the difference fields.
    grid = solver.model.grid
    plotters.plotContourSubplot(uDiff, vDiff, hDiff, 
                                grid.uGrid(), grid.vGrid(), grid.etaGrid(), 
                                model.grid.uField, model.grid.vField, model.grid.hField,
                                levels=20)
    
    plotters.plotContourSubplot(uDiffNormalised, vDiffNormalised, hDiffNormalised, 
                                grid.uGrid(), grid.vGrid(), grid.etaGrid(), 
                                model.grid.uField, model.grid.vField, model.grid.hField)
        
    # Calculate the energy difference between numerical and analytical solutions.
    energyDiff = helpers.calculateEnergy(uDiff, vDiff, hDiff, solver.model.grid.dx, 
                                         solver.model.eqns[0].params)
    
    print(f"Energy difference is {energyDiff:.2e} J")

def runTaskE():
    """ 
    """
    
    dxs = [200e3, 100e3, 50e3, 25e3, 20e3, 10e3, 5e3]
    energyDiffs, energy, timeTaken = [], [], []
    
    # Iterate through different values of dx and calculate energy.
    for i, dxi in enumerate(dxs):
        
        # Create new grid.
        nx = int((xbounds[1] - xbounds[0])/dxi)
        solver.model.grid = ArakawaCGrid(xbounds, nx, periodicX=False)
        
        # Create new timestep from CFL.
        dt = 0.9*helpers.calculateTimestepCFL(100, dxi)
        solver.setNewTimestep(dt, endtime)
        
        # Add the energy equation to be evaluated each time step.
        solver.addCustomEquations("energy", helpers.calculateEnergyModel)
        
        # Record the CPU time to run the solver.
        start = time.process_time()
        solver.run()
        timeTaken.append(time.process_time() - start)
        energy.append(solver.getCustomData("energy"))
        
        # Calculate energy difference at steady state (to analytical).
        energyDiffs.append(helpers.calculateEnergyDifference(solver.model))
    
    # Plots for Task E.
    plotters.plotEnergiesTaskE(dxs, energy, energyDiffs, timeTaken, model, endtime)

def runAndPlotSchemesSteadyState():
    """ 
    """
    # Get the storage arrays for the fields after 40 days.
    hFields, uFields, vFields = helpers.runAllSchemesForNDays(solver, 40)
    
    # Plot the results.
    schemes = ["Forward-backward", "Runge-Kutte-4", "Semi-lagrangian", "Semi-implicit"]
    plotters.plotContoursSchemes(hFields, uFields, vFields, schemes, solver.model.grid)


#### GLOBAL GRID VARIABLES ####
xbounds = [0, 1e6]
dx = 25e3
nx = int((xbounds[1] - xbounds[0])/dx)
grid = ArakawaCGrid(xbounds, nx, periodicX=False)

#### GLOBAL TIME STEPPING VARIABLES ####
dt = 0.85*helpers.calculateTimestepCFL(100, dx)
endtime = 40*24*60**2 
nt = int(np.ceil(endtime/dt))

#### GLOBAL MODEL AND SOLVER ####
model  = Model([Eta(), UVelocity(), VVelocity()], grid)
scheme = helpers.setScheme("forwardBackward")
solver = Solver(model, scheme, dt, nt)

if __name__ == "__main__":
    
    # # Run Task C.    
    # runTaskC()
    
    # # Run Task D.1
    # runTaskD1()
    
    # # Run Task D.2
    # runTaskD2()

    # # Run Task D.3
    # runTaskD3()
    
    # # Run Task E
    # runTaskE()
    
    print("hello")

    #%%
    
    hFields, uFields, vFields = helpers.runAllSchemesForNDays(solver, 40)

    schemes = ["Forward-backward", "Runge-Kutte-4", "Semi-lagrangian", 
              "Semi-implicit"]
    #%% Plot the results.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib as mpl
    
    levels = 50
    fontsize = 15
    ticksize = 12
    figsize = (12, 10)
    
    # Find minimum and maximum values of eta for colorbar.
    hMin, hMax = hFields.min(), hFields.max()
    
    # Calculate analytical solution.
    eta0 = helpers.calculateEta0(hFields[0, ...])
    uSS, vSS, etaSS = helpers.analyticalSolution(solver.model.grid, 
                                                 solver.model.eqns[0].params,
                                                 eta0)
    
    hMin, hMax = (hFields - etaSS).min(), (hFields - etaSS).max()
    
    # Plot the height perturbation contours with streamlines.
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot height perturbation contour plot with streamlines in each subplot
    for i, ax in enumerate(axs.flatten()):
        
        # Get the fields for the ith time scheme.
        hField = hFields[i, ...] - etaSS
        uField = uFields[i, ...] - uSS
        vField = vFields[i, ...] - vSS
        
        # Interpolate the velocity fields onto the eta field.
        uOnEta = 0.5*(uField[:, :-1] + uField[:, 1:])
        vOnEta = 0.5*(vField[:-1, :] + vField[1:, :])
        
        cont = ax.contourf(grid.Xmid/1e3, grid.Ymid/1e3, hField,
                            levels=np.linspace(hMin, hMax, levels)
                            )
        
        # ax.streamplot(grid.Xmid/1000, grid.Ymid/1000, uOnEta, vOnEta, 
        #                 color='black', linewidth=0.5, arrowsize = 0.8,
        #                 density=1.1,)
        ax.set_title(schemes[i], fontsize=fontsize)
        
        if i // 2 != 0:
            ax.set_xlabel("X [km]", fontsize=fontsize)
        else:
            ax.set_xticks([])
        if i % 2 == 0:
            ax.set_ylabel("Y [km]", fontsize=fontsize)
        else:
            ax.set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    # Add a shared colorbar
    cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    plt.colorbar(cont, cax=cax, **kw)
    
    #%%

    plotters.plotAllSchemeSections(grid, hFields, uFields, vFields, 
                                   uSS, vSS, etaSS, schemes)
    
    #%%
    
    # Do this for dx=50km and dx=25km.
    
    s = ["forwardBackward", "rk4", "semiLagrangian", "semiImplicit"]
    timeTaken, energies, energyDiffs = [], [], []
    for i, si in enumerate(s):
        
        # Reset the fields.
        solver.model.grid.resetFields()
        
        # Select the current scheme.
        solver.scheme = helpers.setScheme(si, solver.model, dt)
        
        # Add energy equation to be evaluated at each timestep.
        solver.addCustomEquations("energy", helpers.calculateEnergyModel)
        
        # Run the solver and record the CPU time.
        start = time.process_time()
        solver.run()
        timeTaken.append(time.process_time() - start)
        
        # Get the time evolution of energy.
        energies.append(solver.getCustomData("energy"))
        
        # Calculate the energy difference (to analytical) at steady state.
        energyDiffs.append(helpers.calculateEnergyDifference(solver.model))
                
        # Energy at steady state.
        
        # Time taken. 
#%%

from IPython.display import HTML, display
import tabulate

data = [[si, f"{ei:.2e}J", f"{ti:.2f}s"] for si, ei, ti in zip(schemes, energyDiffs, timeTaken)]
table = tabulate.tabulate(data, tablefmt='html')

time = np.arange(0, solver.dt*(solver.nt + 1), solver.dt)/(24*60**2)
plt.figure(figsize=(10, 10))
for i, energy in enumerate(energies):
    plt.plot(time, energy, label=schemes[i])
plt.grid()
plt.xlabel("Time [days]", fontsize=fontsize)
plt.ylabel("Energy [J]", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.show()

display(HTML(table))

#%% Time stepping stuff.

def varyTimeStepScheme(solver, scheme, dts, endtime, ):
    """ 
    """
    
    timeTaken, energies, energyDiffs = [], [], []
    for i, dti in enumerate(dts):
        
        # Reset the fields.
        solver.model.grid.resetFields()
        
        # Calculate new nt.
        start = time.process_time()
        solver.scheme = helpers.setScheme(scheme, solver.model, dti)
        initSI = time.process_time() - start
                
        solver.setNewTimestep(dti, endtime)
        
        # Add energy equation to be evaluated at each timestep.
        solver.addCustomEquations("energy", helpers.calculateEnergyModel)
        
        # Run the solver and record the CPU time.
        start = time.process_time()
        solver.run()
        timeTaken.append(time.process_time() - start)
        
        # Get the time evolution of energy.
        energies.append(solver.getCustomData("energy"))
        
        # Calculate the energy difference (to analytical) at steady state.
        energyDiffs.append(helpers.calculateEnergyDifference(solver.model))
                
    return energies, energyDiffs, timeTaken, initSI

def plotTimeStepsTaskG(dts, energyDiffs, timeTaken, schemes):
    """ 
    """
    # Plot the things.
    figsize = (15, 6)
    fontsize = 15
    ticksize = 12
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Define the length of a day in seconds for scaling.
    day = 24*60**2
    
    # First subplot: Energy differences vs dts
    for i, si in enumerate(schemes):
        if si == "forwardBackward" or si == "semiLagrangian":
            dtsi = dts[:1]
            markersize=6
        elif si == "rk4":
            dtsi = dts[:2]
            markersize=7
        else:
            dtsi = dts
            markersize=6
        
        axs[0].plot(np.array(dtsi)/day, energyDiffs[i], '-o', label=si, 
                    markersize=markersize)
        axs[1].plot(np.array(dtsi)/day, timeTaken[i], '-o', label=si)
    
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
    axs[1].legend(fontsize=fontsize)
    
    # Add labels to first two dots (less than day so confusing with log).
    axs[1].text((dts[0]+40)/day, timeTaken[2][0]+4, f"$\Delta$t={dts[0]:.0f}s (CFL)",
                    fontsize=0.9*fontsize, rotation=90, ha='right', va='bottom')
    axs[1].text((dts[1] + 75)/day, timeTaken[2][0]+2, f"$\Delta$t={dts[1]:.0f}s",
                    fontsize=0.9*fontsize, rotation=90, ha='right', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
def runTaskG():
    """ 
    """
    day = 24*60**2
    dts = [helpers.calculateTimestepCFL(100, dx), 250, 10e3, day, 5*day, 10*day, 20*day, 40*day]
    
    energies, energyDiffs, timeTaken, initSIs = [], [], [], []    
    schemes = ["semiImplicit", "semiLagrangian", "rk4", "forwardBackward"]
    for i, si in enumerate(schemes):
        if si == "forwardBackward" or si == "semiLagrangian":
            dtsi = dts[:1]
        elif si == "rk4":
            dtsi = dts[:2]
        else:
            dtsi = dts
            
        e, eDiff, t, initSI = varyTimeStepScheme(solver, si, dtsi, endtime)
        
        # Store the current scheme data (lists are eaaasy).
        energies.append(e)
        energyDiffs.append(eDiff)
        timeTaken.append(t)
        initSIs.append(initSI)
        
        # Do something with energies?
    
    print(f"Max semi-implicit inverse matrix construction time: {np.array(initSIs).max()}")
    print(f"Min semi-implicit inverse matrix construction time: {np.array(initSIs).min()}")
    
    # Plot the results.
    plotTimeStepsTaskG(dts, energyDiffs, timeTaken, schemes)
        
runTaskG()