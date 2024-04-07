"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt
import time 
from IPython.display import Image, display

from analyticalSolution import analyticalSolution
from equations import UVelocity, VVelocity, Eta
from grids import ArakawaCGrid
from model import Model
from solver import Solver

import wavesDemo
import plotters
import helpers

def displayArakawaGrid():
    """
    """
    display(Image("arakawaCGrid.png"))

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
    
    # Reset solver fields and run.
    solver.model.grid.resetFields()
    solver.nt = int(np.ceil(day/solver.dt))
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
    solver.nt = int(np.ceil(endtime/solver.dt))
    
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
    
    # Calculate the energy difference between numerical and analytical solutions.
    energyDiff = helpers.calculateEnergy(uDiff, vDiff, hDiff, solver.model.grid.dx, 
                                         solver.model.eqns[0].params)
    
    print(f"Energy difference is {energyDiff:.2e} J")
    
    # Plot the difference fields.
    grid = solver.model.grid
    plotters.plotContourSubplot(uDiff, vDiff, hDiff, 
                                grid.uGrid(), grid.vGrid(), grid.etaGrid(), 
                                model.grid.uField, model.grid.vField, model.grid.hField,
                                levels=20)
    
    plotters.plotContourSubplot(uDiffNormalised, vDiffNormalised, hDiffNormalised, 
                                grid.uGrid(), grid.vGrid(), grid.etaGrid(), 
                                model.grid.uField, model.grid.vField, model.grid.hField)
        

def runTaskE():
    """ 
    """
    
    dxs = [200e3, 100e3, 50e3, 25e3, 20e3, 10e3, 5e3]
    energyDiffs, energy, timeTaken = [], [], []
    
    # Make sure the scheme is forward backward.
    solver.scheme = helpers.setScheme("forwardBackward")
    
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
            
        e, eDiff, t, initSI = helpers.varyTimeStepScheme(solver, si, dtsi, endtime)
        
        # Store the current scheme data (lists are eaaasy).
        energies.append(e)
        energyDiffs.append(eDiff)
        timeTaken.append(t)
        initSIs.append(initSI)
        
        # Do something with energies?
        
    print(f"Max semi-implicit inverse matrix construction time: {np.array(initSIs).max()}")
    print(f"Min semi-implicit inverse matrix construction time: {np.array(initSIs).min()}")
    
    # Plot the results.
    plotters.plotTimeStepsTaskG(dts, energyDiffs, timeTaken, schemes)

def runTaskF1():
    """ 
    """
    plotters.plotHeightContoursRow(hFields[1:, ...], uFields[1:, ...], 
                                   vFields[1:, ...], schemes[1:], gridF, 
                                   solver.model.eqns[0].params)
    
def runTaskF2():
    """ 
    """
    # Calculate the analytical solution.
    eta0 = helpers.calculateEta0(hFields[0, ...])
    uSS, vSS, etaSS = helpers.analyticalSolution(gridF, solver.model.eqns[0].params,
                                                 eta0)
    
    plotters.plotAllSchemeSections(gridF, hFields, uFields, 
                                   vFields, uSS, vSS, etaSS, schemes)
    

def runTaskF3():
    """ 
    """
    
    # Calculate the analytical solution.
    eta0 = helpers.calculateEta0(hFields[0, ...])
    uSS, vSS, etaSS = helpers.analyticalSolution(gridF, solver.model.eqns[0].params,
                                                  eta0)
    
    # Plot the difference.
    uDiffs = uFields - uSS 
    vDiffs = vFields - vSS 
    hDiffs = hFields - etaSS
    

    
    # Plot the differences.
    plotters.plotContoursSchemes(hDiffs, uDiffs, vDiffs, schemes, 
                                 gridF, hFields, uFields, vFields)

#### GLOBAL GRID VARIABLES ####
xbounds = [0, 1e6]
dx = 25e3
nx = int((xbounds[1] - xbounds[0])/dx)
grid = ArakawaCGrid(xbounds, nx, periodicX=False)

#### GLOBAL TIME STEPPING VARIABLES ####
dt = 0.85*helpers.calculateTimestepCFL(100, dx)
day = 24*60**2
endtime = 40*day
nt = int(np.ceil(endtime/dt))

#### GLOBAL MODEL AND SOLVER ####
model  = Model([Eta(), UVelocity(), VVelocity()], grid)
scheme = helpers.setScheme("forwardBackward")
solver = Solver(model, scheme, dt, nt)

# This is done globally because it takes my laptop literal hours on jupyter.
schemes = ["Forward-backward", "Runge-Kutte-4", "Semi-lagrangian", 
          "Semi-implicit"]
hFields, uFields, vFields = helpers.runAllSchemesForNDays(solver, 40)
gridF = solver.model.grid.copy()

#%%
if __name__ == "__main__":
    
    # # Run Task C.    
    # runTaskC()
    
    # Run Task D.1
    runTaskD1()
    
    # Run Task D.2
    runTaskD2()

    # Run Task D.3
    runTaskD3()
    
    # Run Task E
    runTaskE()
    
    # # Run Task G
    # runTaskG()
    
    # runTaskF1()
    
    # runTaskF2()
    
    # runTaskF3()
    
    print("hello")

    #%%
    
#     # hFields, uFields, vFields = helpers.runAllSchemesForNDays(solver, 40)

#     schemes = ["Forward-backward", "Runge-Kutte-4", "Semi-lagrangian", 
#               "Semi-implicit"]
        
#     #%%
    
#     # Do this for dx=50km and dx=25km.
    
#     solver.nt = int(np.ceil(endtime/dt))
    
#     s = ["forwardBackward", "rk4", "semiLagrangian", "semiImplicit"]
#     timeTaken, energies, energyDiffs = [], [], []
#     for i, si in enumerate(s):
        
#         # Reset the fields.
#         solver.model.grid.resetFields()
        
#         # Select the current scheme.
#         solver.scheme = helpers.setScheme(si, solver.model, dt)
        
#         # Add energy equation to be evaluated at each timestep.
#         solver.addCustomEquations("energy", helpers.calculateEnergyModel)
        
#         # Run the solver and record the CPU time.
#         start = time.process_time()
#         solver.run()
#         timeTaken.append(time.process_time() - start)
        
#         # Get the time evolution of energy.
#         energies.append(solver.getCustomData("energy"))
        
#         # Calculate the energy difference (to analytical) at steady state.
#         energyDiffs.append(helpers.calculateEnergyDifference(solver.model))
                
# #%%

#     from IPython.display import HTML, display
#     import tabulate
    
#     data = [[si, f"{ei:.2e}J", f"{ti:.2f}s"] for si, ei, ti in zip(schemes, energyDiffs, timeTaken)]
#     table = tabulate.tabulate(data, tablefmt='html')
        
#     display(HTML(table))