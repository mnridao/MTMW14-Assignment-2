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
from timeSchemes import SemiImplicitSchemeCoupled
from plotters import plotContourSubplot

def calculateEnergy(model):
    """ 
    """
    params = model.eqns[0].params
    u = model.grid.uField
    v = model.grid.vField
    eta = model.grid.hField
    
    term1 = params.H*(np.sum(u**2) + np.sum(v**2))
    term2 = params.g*np.sum(eta**2)
    return 0.5*params.rho*model.grid.dx*model.grid.dy*(term1+term2)

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
    dt = 0.5*calculateTimestepCFL(100, dx)
    # dt = 10*24*60**2
    endtime = 50*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    # Set up the model and solver.
    # scheme = RK4SchemeCoupled
    scheme = forwardBackwardSchemeCoupled
    # scheme = SemiLagrangianSchemeCoupled()
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    
    # scheme = SemiImplicitSchemeCoupled(model, dt)
    
    solver = Solver(model, scheme, dt, nt)
    
    # Add energy calculator to solver.
    solver.addCustomEquations("energy", calculateEnergy)
        
    #%% Semi implicit class test
    solver.model.grid.resetFields()
    
    solver.model.activateWindStress(False)
    solver.model.activateDamping(False)
    solver.model.setf0(0)
    solver.model.setBeta(0)
    
    solver.model.setBlobInitialCondition(xL*np.array([0.1, 0.5]), 
                                          ((3*dx)**2*np.array([2, 2])**2), 1*dx)
    # plotContourSubplot(solver.model.grid)
    
    solver.run()
    
    #%% Task D (get plots working here)
    # solver.store = True
    solver.run()
    energy = solver.getCustomData("energy")
    # plotContourSubplot(solver.model.grid)
    
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
    # plt.plot(time, energy, label="$\Delta x$=50km")
    # plt.plot(timeHalf, energyHalf, label="$\Delta x$=25km")
    plt.grid()
    plt.legend()
    plt.show()
    
    #%% Sea mountain + gravity wave.
    solver.model.grid.resetFields()
    
    solver.model.activateWindStress(False)
    solver.model.setf0(0)
    solver.model.setBeta(0)
    
    mu = xL*np.array([0.5, 0.55])
    var = (1*dx)**2*np.array([2, 2])**2
    solver.model.setMountainBottomTopography(mu, var, 100)
    solver.model.setBlobInitialCondition(np.array([0, 0]), var, 10)
    
    plotContourSubplot(solver.model.grid)
    
    solver.store = True
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Gravity wave with blob initial condition.
    solver.model.grid.resetFields()
    
    solver.model.activateWindStress(False)
    solver.model.activateDamping(False)
    solver.model.setf0(0)
    solver.model.setBeta(0)
    
    solver.model.setBlobInitialCondition(xL*np.array([0.5, 0.55]), 
                                          ((3*dx)**2*np.array([2, 2])**2), 1*dx)
    plotContourSubplot(solver.model.grid)
    
    solver.store = True
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

    # # Create new grid for equatorial beta plate.
    # # grid = ArakawaCGrid(xbounds, nx, [-0.5*xL, 0.5*xL], periodicX=True)

    # # Equatorial beta plane.
    # solver.model.setf0(f0)
    # solver.model.setBeta(beta)   # Increase the effects of rotation.
    # # solver.model.grid = grid
    
    # Add a mountain.
    solver.model.setMountainBottomTopography(xL*np.array([0.5, 0.55]), 
                                          ((1*dx)**2*np.array([2, 2])**2), 900)
    
    # Set up equatorial easterly.
    Y = solver.model.grid.Ymid
    # f = (f0 + beta*(Y - Y.mean()))
    f = f0 + beta*Y

    # solver.model.grid.hField = 1000. - 50.*np.cos((Y-np.mean(Y))*4.*np.pi/np.max(Y));
    mean_wind_speed = 20.; # m/s
    # solver.model.grid.hField = 1000.-(mean_wind_speed*f/g)*(Y-np.mean(Y)); 
    solver.model.grid.hField = 1000.-(mean_wind_speed*f/g)*Y - solver.model.grid.hBot; 
    
    # Update the viewer.
    solver.model.grid.fields["eta"] = solver.model.grid.hField
    
    # Setup geostrophic wind.
    solver.model.setGeostrophicBalanceInitialCondition()
    
    solver.run()
    
    plotContourSubplot(solver.model.grid)
    
    #%% Rossby wave attempt no 2.
    
    solver.model.grid.resetFields()
    solver.store = True
    
    solver.model.activateWindStress(False)
    
    # Create new grid for equatorial beta plate.
    grid = ArakawaCGrid(xbounds, nx, [-0.5*xL, 0.5*xL], periodicX=True)

    # Equatorial beta plane.
    solver.model.setf0(1e-5)
    solver.model.setBeta(1e-11)   # Increase the effects of rotation.
    solver.model.grid = grid
    
    # blob initial condition.
    solver.model.setBlobInitialCondition(xL*np.array([0.45, 0.55]), 
                                          ((2*dx)**2*np.array([2, 2])**2), 100)
    
    # Set bottom topography.
    solver.model.setMountainBottomTopography(xL*np.array([0.5, 0.55]), 
                                          ((2*dx)**2*np.array([2, 2])**2), 100)
    
    # Set geostrophic initial conditions.
    solver.model.setGeostrophicBalanceInitialCondition()
    solver.run()
    plotContourSubplot(solver.model.grid)
    
    #%% Trying out different plots.
    solver.model.grid.resetFields()
    
    solver.store = True
    solver.model.setBlobInitialCondition(xL*np.array([0.5, 0.55]), 
                                          ((5*dx)**2*np.array([2, 2])**2), 100)
    # plotContourSubplot(solver.model.grid)
    # solver.run()
    
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
    
    minH = min(np.min(state[2]) for state in solver.history)
    maxH = max(np.max(state[2]) for state in solver.history)
    
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
    
        state = states[frame]
    
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
    states = solver.history
    total_frames = len(states)
        
    # Use FuncAnimation to create the animation
    animation = FuncAnimation(fig, update, frames=total_frames
                               , interval=200
                              )
    
    plt.show()
    
    # Save the animation as a GIF
    from matplotlib.animation import PillowWriter
    
    writer = PillowWriter(fps=30)
    animation.save("gravityWaveFB.gif", writer=writer)
    
    # Close the figure to avoid displaying it twice
    plt.close(fig)