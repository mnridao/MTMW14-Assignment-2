"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt

from solver import Solver
from model import Model
from equations import UVelocity, VVelocity, Eta
from grids import ArakawaCGrid
from timeSchemes import forwardBackwardSchemeCoupled, RK4SchemeCoupled, SemiLagrangianSchemeCoupled
from plotters import plotContourSubplot

def calculateEnergy(model):
    """ 
    """
    params = model.eqns[0].params
    return (np.sum(0.5*params.rho*(model.grid.uField[:, :-1]**2 + 
                                   model.grid.vField[:-1, :]**2 + 
                                   params.g*model.grid.hField**2))* 
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
    # nx = 254
    grid = ArakawaCGrid(xbounds, nx, periodicX=False)

    # Time stepping information.
    dt = 0.99*calculateTimestepCFL(100, dx)
    # dt = 1000
    # dt = 100
    endtime = 30*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    # Set up the model and solver.
    scheme = RK4SchemeCoupled
    # scheme = SemiLagrangianSchemeCoupled()
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    solver = Solver(model, scheme, dt, nt)
    
    # Add energy calculator to solver.
    solver.addCustomEquations("energy", calculateEnergy)
        
    #%% Task D (get plots working here)
    solver.run()
    # energy = solver.getCustomData("energy")
    plotContourSubplot(solver.model.grid)
    
    # Plot energy.
    time = np.arange(0, solver.dt*(solver.nt+1), solver.dt)
    plt.figure(figsize=(10, 10))
    # plt.plot(energy)
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
    # plt.plot(time, energy, label="$\Delta x$=50km")
    # plt.plot(timeHalf, energyHalf, label="$\Delta x$=25km")
    plt.grid()
    plt.legend()
    plt.show()
            
    #%% Gravity wave with blob initial condition.
    solver.model.grid.resetFields()
    
    solver.model.activateWindStress(False)
    solver.model.setH(1e3)
    solver.model.setf0(0)
    solver.model.setBeta(0)
    
    solver.model.setBlobInitialCondition(xL*np.array([0.5, 0.55]), 
                                          ((1*dx)**2*np.array([2, 2])**2), 0.01*dx)
    plotContourSubplot(solver.model.grid)
    
    # solver.store = True
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Kelvin wave attempt (increase beta?).
    solver.model.grid.resetFields()
    solver.store = True
    
    solver.model.activateWindStress(False)
    
    # Create new grid for equatorial beta plate.
    grid = ArakawaCGrid(xbounds, nx, [-0.5*xL, 0.5*xL], periodicX=True)

    # Equatorial beta plane.
    solver.model.setf0(0)
    solver.model.setBeta(5e-8)   # Increase the effects of rotation.
    solver.model.grid = grid
    
    solver.model.setBlobInitialCondition(np.array([0.2*xL, 0]), 
                                          ((5*dx)**2*np.array([2, 2])**2), 100)
    
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Rossby wave attempt (using easterly jet initial condition).
    
    # THis has become baroclinic instability attempt which also doesn't wok
    
    f0 = 1e-4
    # f0 = 0.
    beta = 1.6e-11
    # beta = 2e-11
    g = 9.8
    
    solver.model.activateWindStress(False)
    
    solver.model.grid.resetFields()
    solver.store = True
    
    solver.model.setSharpShearInitialCondition()
    
    solver.run()
    
    plotContourSubplot(solver.model.grid)
    #%% Trying out different plots.
    solver.model.grid.resetFields()
    
    solver.store = True
    solver.model.setBlobInitialCondition(xL*np.array([0.5, 0.55]), 
                                          ((5*dx)**2*np.array([2, 2])**2), 100)
    plotContourSubplot(solver.model.grid)
    solver.run()
    
    plotContourSubplot(solver.model.grid)
    
    #%% Plot velocity quiver plots.
    for state in solver.history:
    
        fig, ax = plt.subplots(figsize = (8, 8), facecolor = "white")
        plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontsize = 19)
        plt.xlabel("x [km]", fontsize = 16)
        plt.ylabel("y [km]", fontsize = 16)
        q_int = 3
        Q = ax.quiver(solver.model.grid.X[::q_int, ::q_int]/1000.0, 
                      solver.model.grid.Y[::q_int, ::q_int]/1000.0, 
                      state[0][::q_int,::q_int], 
                      state[1][::q_int,::q_int],
            scale=10, scale_units='inches')
        plt.show()
        
    #%% Plot height perturbation contour plots.
    from matplotlib import colors
    
    minH = min(np.min(state[2]) for state in solver.history)
    maxH = max(np.max(state[2]) for state in solver.history)
        
    for state in solver.history:
        plt.figure(figsize=(10, 10))
        
        levels = 100
        
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
    
    minH = min(np.min(state[2]) for state in solver.history[::100])
    maxH = max(np.max(state[2]) for state in solver.history[::100])
    
    for state in solver.history:
        
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        
        # norm = colors.Normalize(vmin=minH, vmax=maxH)
        
        # Plot the surface
        surf = ax.plot_surface(solver.model.grid.Xmid, 
                               solver.model.grid.Ymid, 
                               state[2], 
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
    
    minH = min(np.min(state[2]) for state in solver.history)
    maxH = max(np.max(state[2]) for state in solver.history)
    
    # Set up the figure and axis
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
    
        state = solver.history[frame]
    
        # Plot the surface
        surf = ax.plot_surface(solver.model.grid.Xmid,
                                solver.model.grid.Ymid,
                                state[2],
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
    total_frames = len(solver.history)
    
    # Use FuncAnimation to create the animation
    animation = FuncAnimation(fig, update, frames=total_frames
                               , interval=200
                              )
    
    plt.show()
    
    # Save the animation as a GIF
    from matplotlib.animation import PillowWriter
    
    writer = PillowWriter(fps=30)
    animation.save("kelvinWavePeriodic2.gif", writer=writer)
    
    # Close the figure to avoid displaying it twice
    plt.close(fig)
