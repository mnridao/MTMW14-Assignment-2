"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt

from solver import Solver, Model
from equations import UVelocity, VVelocity, Eta
from grids import ArakawaCGrid
from timeSchemes import forwardBackwardSchemeCoupled
from plotters import plotContourSubplot

def createGrid(xbounds, dx, ybounds, d):
    """ 
    """
    pass

def calculateEnergy(model):
    """ 
    """
    params = model.eqns[0].params
    return (np.sum(0.5*params.rho*(model.grid.uField[:, :-1]**2 + 
                                  model.grid.vField[:-1, :]**2 + 
                                  params.g*model.grid.hField**2)) * 
            model.grid.dx**2)

def calculateTimestepCFL(c, d):
    """ 
    """
    return np.floor(d/(c*np.sqrt(2)))

if __name__ == "__main__":
        
    # Grid creation.
    xbounds = [0, 1e6]
    xL = xbounds[1]
    dx = 50e3
    nx = int((xbounds[1] - xbounds[0])/dx)
    grid = ArakawaCGrid(xbounds, nx)
    
    # Time stepping information.
    dt = calculateTimestepCFL(100, dx) - 3
    endtime = 50*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    # Set up the model and solver.
    scheme = forwardBackwardSchemeCoupled
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    solver = Solver(model, scheme, dt, nt)
    
    # Add energy calculator to solver.
    solver.addCustomEquations("energy", calculateEnergy)
        
    #%% Task D (get plots working here)
    solver.run()
    energy = solver.getCustomData("energy")
    plotContourSubplot(solver.model.grid)
    
    # Plot energy.
    plt.figure(figsize=(10, 10))
    plt.plot(energy)
    plt.show()
    
    # Quiver plot for velocity.
    # fig, ax = plt.subplots(figsize = (8, 8), facecolor = "white")
    # plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 19)
    # plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
    # plt.ylabel("y [km]", fontname = "serif", fontsize = 16)
    # q_int = 3
    # Q = ax.quiver(grid.X[::q_int, ::q_int]/1000.0, grid.Y[::q_int, ::q_int]/1000.0, solver.model.grid.uField[::q_int,::q_int], solver.model.grid.vField[::q_int,::q_int],
    #     scale=0.2, scale_units='inches')
    # plt.show()
    
    # Height perturbation plot.
    
    # Height perturbation plot 3D.
    
    #%% Task E (energy)
    
    # Half the grid spacing and find new time step.
    solver.model.grid = ArakawaCGrid(xbounds, nx*2)
    
    dt = calculateTimestepCFL(100, solver.model.grid.dx)
    solver.setNewTimestep(dt, endtime)
            
    # Run the solver and plot the new energy.
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    energyHalf = solver.getCustomData("energy")
    
    # time = np.arange(0, solver.nt*(solver.dt+1), solver.dt)
    plt.figure(figsize=(10, 10))
    plt.plot(energy, label="$\Delta x$=50km")
    plt.plot(energyHalf, label="$\Delta x$=25km")
    plt.grid()
    plt.legend()
    plt.show()
    
    #%% Turn rotation on/off.

    solver.model.activateBetaPlane(False)
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Turn wind on/off.
    solver.model.activateWindStress(True)
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Gravity wave with step initial condition.
    solver.model.activateBetaPlane(False)
    solver.model.activateWindStress(False)
    
    solver.model.setStepInitialCondition(xL*np.array([0.5, 0.55]), 
                                         xL*np.array([0.5, 0.55]), 100*dx)
    plotContourSubplot(solver.model.grid)
    
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Gravity wave with blob initial condition.
    solver.model.activateBetaPlane(False)
    solver.model.activateWindStress(False)
    
    solver.model.setBlobInitialCondition(xL*np.array([0, 0.5]), 
                                         (xL*np.array([0.05, 0.05]))**2, 1e6*dx)
    # plotContourSubplot(solver.model.grid)
    
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Kelvin wave attempt (increase beta?).
    solver.model.activateBetaPlane(True)
    solver.model.activateWindStress(False)
    
    # Create new grid for equatorial beta plate.
    grid = ArakawaCGrid(xbounds, nx, [-0.5*xL, 0.5*xL])
    
    # Equatorial beta plane.
    solver.model.setf0(0)
    solver.model.setBeta(5e-8)   # Increase the effects of rotation.
    solver.model.grid = grid
    
    solver.model.setStepInitialCondition(xL*np.array([0., 0.05]), 
                                         xL*np.array([0.5, 0.55]), 100*dx)
    
    solver.run()
    plotContourSubplot(solver.model.grid)