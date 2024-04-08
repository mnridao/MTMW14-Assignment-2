"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import time 
from IPython.display import Image, HTML, display
import tabulate

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
                                grid.uGrid(), grid.vGrid(), grid.etaGrid(),
                                levels=25)

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
            
    # Calculate the analytical solution.
    eta0 = helpers.calculateEta0(solver.model.grid.hField)
    uSS, vSS, etaSS = analyticalSolution(solver.model.grid, 
                                         solver.model.eqns[0].params, eta0)
    
    # Calculate the difference fields.    
    uDiff = solver.model.grid.uField - uSS
    vDiff = solver.model.grid.vField - vSS
    hDiff = solver.model.grid.hField - etaSS
    
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
                                solver.model.grid.uField, 
                                solver.model.grid.vField, 
                                solver.model.grid.hField, levels=20)
    
    plotters.plotContourSubplot(uDiffNormalised, vDiffNormalised, hDiffNormalised, 
                                grid.uGrid(), grid.vGrid(), grid.etaGrid(), 
                                solver.model.grid.uField, 
                                solver.model.grid.vField, 
                                solver.model.grid.hField)
        

def runTaskE():
    """ 
    Calculates the energy time series, energy differences and CPU time taken 
    to run the solver for a range of different grid spacings. This takes a 
    while to run, on my laptop it takes a few minutes.
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
    Calculates the energy differences and CPU time taken to run the solver for 
    a range of different time steps, and for all the different numerical 
    schemes implemented in this project.
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
   
    # Plot the results.
    plotters.plotTimeStepsTaskG(dts, energyDiffs, timeTaken, schemes)

def runTaskF1():
    """ 
    Plots the steady state distribution of the gyre from the non-linear 
    semi-lagrangian model.
    """
        
    # Get the semi lagrangian solutions for 40 days.
    hField = hFields[2, ...]
    uField = uFields[2, ...]
    vField = vFields[2, ...]
    
    # Calculate the analytical solution for the linear model.
    eta0 = helpers.calculateEta0(hField)
    uSS, vSS, etaSS = analyticalSolution(gridF, 
                                         solver.model.eqns[0].params, eta0)
    
    # Plot steady state distribution of gyres.
    plotters.plotContourSubplot(uField, vField, hField, 
                                gridF.uGrid(), gridF.vGrid(), gridF.etaGrid(),
                                uSS, vSS, etaSS,
                                levels=25)
    
def runTaskF2():
    """ 
    Plots the semi-lagrangian solution at different cross sections of the gyre.
    """
    
    # Get the semi lagrangian solutions for 40 days.
    hField = hFields[2, ...]
    uField = uFields[2, ...]
    vField = vFields[2, ...]
    scheme = schemes[2]
    
    # Calculate the analytical solution for the linear model.
    eta0 = helpers.calculateEta0(hField)
    uSS, vSS, etaSS = analyticalSolution(gridF, 
                                         solver.model.eqns[0].params, eta0)
    
    # Plot the sections
    plotters.plotAllSchemeSections(gridF, np.expand_dims(hField, 0), 
                                   np.expand_dims(uField, 0), 
                                   np.expand_dims(vField, 0), 
                                   uSS, vSS, etaSS, [scheme])
    

def runTaskF3():
    """ 
    Plots contours of the field differences for the semi-lagrangian numerical 
    solution.
    """
    
    # Get the semi lagrangian solutions for 40 days.
    hField = hFields[2, ...]
    uField = uFields[2, ...]
    vField = vFields[2, ...]
    
    # Calculate the analytical solution for the linear model.
    eta0 = helpers.calculateEta0(hField)
    uSS, vSS, etaSS = analyticalSolution(gridF, 
                                         solver.model.eqns[0].params, eta0)
    
    # Plot the energy difference.
    uDiffs = uField - uSS 
    vDiffs = vField - vSS 
    hDiffs = hField - etaSS
    
    # Calculate the energy difference between numerical and analytical solutions.
    energyDiff = helpers.calculateEnergy(uDiffs, vDiffs, hDiffs, gridF.dx, 
                                         solver.model.eqns[0].params)
    
    print(f"Energy difference is {energyDiff:.2e} J")
        
    plotters.plotContourSubplot(uDiffs, vDiffs, hDiffs, 
                                gridF.uGrid(), gridF.vGrid(), gridF.etaGrid(),
                                uField, vField, hField,
                                levels=25)

def runTaskG1():
    """ 
    
    """
    plotters.plotHeightContoursRow(hFields[[0, 1, 3], ...], 
                                   uFields[[0, 1, 3], ...], 
                                   vFields[[0, 1, 3], ...], 
                                   ["Forward-backward", "Runge-Kutte-4", 
                                    "Semi-implicit"], 
                                   gridF, 
                                   solver.model.eqns[0].params)
    
def runTaskG2():
    """ 
    """
    # Calculate the analytical solution.
    eta0 = helpers.calculateEta0(hFields[0, ...])
    uSS, vSS, etaSS = helpers.analyticalSolution(gridF, solver.model.eqns[0].params,
                                                  eta0)
    
    plotters.plotAllSchemeSections(gridF, hFields, uFields, 
                                    vFields, uSS, vSS, etaSS, schemes)
    

def runTaskG3():
    """ 
    """
    
    # Calculate the analytical solution.
    eta0 = helpers.calculateEta0(hFields[0, ...])
    uSS, vSS, etaSS = helpers.analyticalSolution(gridF, solver.model.eqns[0].params,
                                                  eta0)
    
    hFields2 = hFields[[0, 1, 3], ...]
    uFields2 = uFields[[0, 1, 3], ...] 
    vFields2 = vFields[[0, 1, 3], ...]
    
    # Plot the difference.
    uDiffs = uFields2 - uSS 
    vDiffs = vFields2 - vSS 
    hDiffs = hFields2 - etaSS    
    
    # Plot the differences.
    schemes2 = ["Forward-backward", "Runge-Kutte-4", "Semi-implicit"]
    plotters.plotContoursSchemes(hDiffs, uDiffs, vDiffs, schemes2, 
                                  gridF, hFields2, uFields2, vFields2)

def runTaskGTable():
    """ 
    """
    
    # Initialise energy difference array.
    energyDiffs = np.zeros(shape=len(schemes))
    
    for i, s in enumerate(schemes):
        
        # Calculate the analytical solution.
        eta0 = helpers.calculateEta0(hFields[i, ...])
        uSS, vSS, etaSS = helpers.analyticalSolution(gridF, solver.model.eqns[0].params,
                                                      eta0)        
        # Plot the difference fields.
        uDiffs = uFields[i, ...] - uSS 
        vDiffs = vFields[i, ...] - vSS 
        hDiffs = hFields[i, ...] - etaSS    
        
        # Calculate the energy difference.
        energyDiffs[i] = helpers.calculateEnergy(uDiffs, vDiffs, hDiffs, 
                                                 gridF.dx, 
                                                 solver.model.eqns[0].params)
    
    data = [[si, f"{ei:.2e}J", f"{ti:.2f}s"] for si, ei, ti in zip(schemes, energyDiffs, timeTaken)]
    
    # Add a header.
    header = ["Scheme", "Energy Difference", "Time Taken"]
    data.insert(0, header)
    
    table = tabulate.tabulate(data, tablefmt='html')
        
    display(HTML(table))
    
def runTaskG4():
    """ 
    """
    
    # Set the timestep to a day.
    solver.setNewTimestep(day, endtime)
    
    # Set the scheme to semi-implicit and run for a day.
    start = time.process_time()
    solver.scheme = helpers.setScheme("semiImplicit", solver.model, day)
    end = time.process_time() - start 
    
    print(f"Inverse matrix construction time: {end}s")
    
    # Reset the fields and run the solver.
    solver.model.grid.resetFields()
    
    start = time.process_time()
    solver.run()
    end = time.process_time() - start
    
    print(f"CPU time: {end}s")
    
    # Plot sections.
    plotters.plotSteadyStateWithAnalytical(solver.model)
        
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
hFields, uFields, vFields, timeTaken = helpers.runAllSchemesForNDays(solver, 40)
gridF = solver.model.grid.copy()

#%%
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
    
    # # Run Task G
    # runTaskG()
    
    # # Run Task F
    # runTaskF1()
    # runTaskF2()
    # runTaskF3()
    
    # # Run Task G
    # runTaskG1()
    # runTaskG2()
    # runTaskG3()
    # runTaskGTable()
    # runTaskG()
    # runTaskG4()
    
    print("hello")