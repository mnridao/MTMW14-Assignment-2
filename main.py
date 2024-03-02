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
    dx = 25e3
    nx = int((xbounds[1] - xbounds[0])/dx)
    grid = ArakawaCGrid(xbounds, nx)
    
    # Time stepping information.
    dt = calculateTimestepCFL(100, dx) - 3
    endtime = 10*24*60**2 
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
    time = np.arange(0, solver.dt*(solver.nt+1), solver.dt)
    plt.figure(figsize=(10, 10))
    plt.plot(energy)
    plt.show()
        
    #%% Task E (energy)
    
    # Half the grid spacing and find new time step.
    solver.model.grid = ArakawaCGrid(xbounds, nx*2)
    
    dt = calculateTimestepCFL(100, solver.model.grid.dx)
    solver.setNewTimestep(dt, endtime)
            
    # Run the solver and plot the new energy.
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    energyHalf = solver.getCustomData("energy")
    
    timeHalf = np.arange(0, solver.dt*(solver.nt+1), solver.dt)
    plt.figure(figsize=(10, 10))
    plt.plot(time, energy, label="$\Delta x$=50km")
    plt.plot(timeHalf, energyHalf, label="$\Delta x$=25km")
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
    plotContourSubplot(solver.model.grid)
    
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
    
    
    #%% Trying out different plots.
    solver.model.grid.resetFields()
    
    solver.store = True
    # solver.model.setStepInitialCondition(xL*np.array([0.5, 0.55]), 
    #                                        xL*np.array([0.5, 0.55]), 50*dx)
    solver.model.setBlobInitialCondition(xL*np.array([0.5, 0.55]), 
                                          (dx**2*np.array([2, 2])**2), 2*dx)
    plotContourSubplot(solver.model.grid)
    solver.run()
    
    plotContourSubplot(solver.model.grid)
    
    #%% Plot velocity quiver plots.
    for state in solver.history:
    
        fig, ax = plt.subplots(figsize = (8, 8), facecolor = "white")
        plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 19)
        plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
        plt.ylabel("y [km]", fontname = "serif", fontsize = 16)
        q_int = 3
        Q = ax.quiver(solver.model.grid.X[::q_int, ::q_int]/1000.0, 
                      solver.model.grid.Y[::q_int, ::q_int]/1000.0, 
                      state[0][::q_int,::q_int], 
                      state[1][::q_int,::q_int],
            scale=50, scale_units='inches')
        plt.show()
        
    #%% Plot height perturbation contour plots.
    from matplotlib import colors
    
    minH = min(np.min(state[2]) for state in solver.history)
    maxH = max(np.max(state[2]) for state in solver.history)
        
    for state in solver.history:
        plt.figure(figsize=(10, 10))
        
        levels = 75
        
        cmap = plt.get_cmap('viridis', levels)

        # Normalize the data based on minH and maxH
        norm = colors.Normalize(vmin=minH, vmax=maxH)
        
        cont = plt.contourf(solver.model.grid.X[:-1, :-1], 
                            solver.model.grid.Y[:-1,:-1], 
                            state[2], 
                            # vmin=minH, vmax=maxH, 
                            levels=levels, cmap=cmap, norm=norm)

        plt.show()
        # break
    
    #%% Plot heigh perturbation surface plots.
    
    minH = min(np.min(state[2]) for state in solver.history)
    maxH = max(np.max(state[2]) for state in solver.history)
    
    # Assuming solver.history is a list of states
    for state in solver.history:
        
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        
        # norm = colors.Normalize(vmin=minH, vmax=maxH)
        
        # Plot the surface
        surf = ax.plot_surface(solver.model.grid.X[:-1, :-1], 
                               solver.model.grid.Y[:-1, :-1], 
                               state[2], 
                                cmap='viridis', 
                               rstride=5, cstride=5, antialiased=True,
                                vmin=minH, vmax=maxH
                               )
        
        ax.set_zlim(-5, 3*dx)
        
        # Customize the plot
        ax.set_xlabel('X [km]', fontsize=25)
        ax.set_ylabel('Y [km]', fontsize=25)
        # ax.set_zlabel('Z')
    
        plt.show()
