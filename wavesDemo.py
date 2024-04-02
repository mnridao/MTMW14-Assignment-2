"""
MTMW14 Assignment 2

Student ID: 31827379
"""
import numpy as np
from IPython.display import HTML, display

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
from functools import partial

from equations import UVelocity, VVelocity, Eta
from grids import ArakawaCGrid
from model import Model
from solver import Solver
import helpers

#### GLOBAL WAVE DEMO VARIABLES ####
xbounds = [0, 1e6]
dx = 5e3
nx = int((xbounds[1] - xbounds[0])/dx)
scheme = helpers.setScheme("rk4")
dt = 0.99*helpers.calculateTimestepCFL(100, dx)
nt = 500

grid = ArakawaCGrid(xbounds, nx, periodicX=False)
model = Model([Eta(), UVelocity(), VVelocity()], grid)
solver = Solver(model, scheme, dt, nt)

# Remove wind stress and damping.
solver.model.activateWindStress(False)
solver.model.activateDamping(False)

def displayGravityWave():
    """ 
    """
    
    htmlTable = """<style>.gif-table {
    width: 90%;
    table-layout: fixed;
    border-collapse: collapse;
    }
    .gif-table td {
    padding: 0;
    text-align: center;
    width: 90%;
    }
    .img-container {
    display: inline-block;
    width: 100%;
    text-align: center;
    margin: 0;
    }
    .gif-table img {
    max-width: 100%;
    height: auto;
    }
    .center-table {
    margin: 0 auto;
    margin-left: 10%;
    }
    </style>
    <div class="center-table">
    <table class='gif-table'>
    <tr>
        <td>
            <div class="img-container"><img src='gravityWaveEta.gif'></div>
        </td>
        <td>
            <div class="img-container"><img src='gravityWaveVelocity.gif'></div>
        </td>
    </tr>
    </table>
    </div>
    """
    
    display(HTML(htmlTable))
    
def displayKelvinWave():
    """ 
    """
    htmlTable = """<style>.gif-table {
    width: 90%;
    table-layout: fixed;
    border-collapse: collapse;
    }
    .gif-table td {
    padding: 0;
    text-align: center;
    width: 90%;
    }
    .img-container {
    display: inline-block;
    width: 100%;
    text-align: center;
    margin: 0; /* Remove margin */
    }
    .gif-table img {
    max-width: 100%;
    height: auto;
    }
    .center-table {
    margin: 0 auto;
    margin-left: 10%;
    }
    </style>
    <div class="center-table">
    <table class='gif-table'>
    <tr>
        <td>
            <div class="img-container"><img src='kelvinWaveEta.gif'></div>
        </td>
        <td>
            <div class="img-container"><img src='kelvinWaveVelocity.gif'></div>
        </td>
    </tr>
    </table>
    </div>
    """
    
    display(HTML(htmlTable))

def updateSurfaceElevation(frame, ax, X, Y, states):
    """ 
    Updates the surface elevation for the current frame.
    
    Inputs
    ------
    frame  : int 
             The current frame of the animation.
    ax     : matplotlib Axes object
             The axis of the figure that is being updated.
    X      : np array
             X-domain coordinates.
    Y      : np array
             Y-domain coordinates.
    states : list of np arrays
             All arrays of surface elevation for each frame.
    """
    
    # Find max and min eta values to fix the z axis.
    minH = min(np.min(state[2]) for state in states)
    maxH = max(np.max(state[2]) for state in states)
    
    # Clear the previous plot.
    ax.clear()
    
    # Plot the surface elevation.
    surf = ax.plot_surface(X/1000, Y/1000, states[frame][2], cmap='viridis',
                           rstride=5, cstride=5, antialiased=True,
                           vmin=minH, vmax=maxH
                           )

    ax.tick_params(axis="both", which="both", labelsize=20)
    
    # Make sure the background is white (default is grey).
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_zlim(minH, maxH)
    ax.set_xlim(X.min()/1000, X.max()/1000)
    ax.set_ylim(Y.min()/1000, Y.max()/1000)
    ax.set_xlabel('X [km]', fontsize=25)
    ax.set_ylabel('Y [km]', fontsize=25)
        
    return surf,

def updateVelocityField(frame, ax, X, Y, states, scale=1):
    """ 
    Updates the velocity field plot for the current frame.
    
    Inputs
    ------
    frame  : int 
             The current frame of the animation.
    ax     : matplotlib Axes object
             The axis of the figure that is being updated.
    X      : np array
             X-domain coordinates.
    Y      : np array
             Y-domain coordinates.
    states : list of np arrays
             All arrays of surface elevation for each frame.
    scale  : Arrowhead scale. Default is 1.
    """
    
    # Clear the previous plot.
    ax.clear()
    
    ax.set_xlabel("x [km]", fontsize=25)
    ax.set_ylabel("y [km]", fontsize=25)
    q_int = 6
    Q = ax.quiver(solver.model.grid.X[::q_int, ::q_int] / 1000.0,
                  solver.model.grid.Y[::q_int, ::q_int] / 1000.0,
                  states[frame][0][::q_int, ::q_int],
                  states[frame][1][::q_int, ::q_int],
                  scale=scale, scale_units='inches')
    ax.tick_params(axis="both", which="both", labelsize=20)

    return Q,

def animateSurfaceElevation(solver, filename):
    """
    Creates an animation of the surface elevation and saves it as a gif.
    
    Inputs
    ------
    solver   : Solver object
               Object containing the model and the states.
    filename : string
               path + filename for gif file.
    """
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use FuncAnimation to create the animation
    animation = FuncAnimation(fig, partial(updateSurfaceElevation, ax=ax, 
                                            X=solver.model.grid.Xmid, 
                                            Y=solver.model.grid.Ymid, 
                                            states=solver.history), 
                              frames=len(solver.history), 
                              interval=200
                              )
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    plt.show()
    
    # Save the animation as a GIF
    writer = PillowWriter(fps=30)
    animation.save(filename, writer=writer)

def animateVelocityField(solver, arrowScale, filename):
    """
    Creates an animation of the velocity field and saves it as a gif.
    
    Inputs
    ------
    solver   : Solver object
               Object containing the model and the states.
    scale    : int
               Arrowhead scale for the quiver plot.
    filename : string
               path + filename for gif file.
    """
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    
    animation = FuncAnimation(fig, partial(updateVelocityField, ax=ax, 
                                            X=solver.model.grid.X, 
                                            Y=solver.model.grid.Y, 
                                            states=solver.history), 
                              frames=len(solver.history),
                              interval = 200)
    
    writer = PillowWriter(fps=30)
    animation.save(filename, writer=writer)

def runAndAnimateGravityWaves():
    """
    Run a simulation showing a gravity wave and save the animation as a gif.
    
    Inputs
    ------
    solver : Solver object
             Solver with default grid and model.
    """
    
    solver.model.grid.resetFields()
    
    # Remove rotation.
    solver.model.setf0(0)
    solver.model.setBeta(0)
    
    # Set initial blob condition.
    mu = xbounds[1]*np.array([0.3, 0.3])
    var = (5*dx)**2*np.array([2, 2])**2
    solver.model.setBlobInitialCondition(mu, var, 0.01*dx)
    
    # Run the solver and store every time step.
    solver.store = True
    solver.run()
        
    # Create animation of surface elavation (eta).
    animateSurfaceElevation(solver, "gravityWaveEta.gif")
    
    # Create animation of velocity field.
    animateVelocityField(solver, 3, "gravityWaveVelocity.gif")

def runAndAnimateKelvinWaves():
    """
    Run a simulation showing a gravity wave and save the animation as a gif.
    
    Inputs
    ------
    solver : Solver object
             Solver with default grid and model.
    """
    solver.model.grid.resetFields()
    
    # Adjust the equation parameters for equator.
    solver.model.setf0(0)
    solver.model.setBeta(1e-8)   # Increase the effects of rotation.
    
    # Create grid centred at equator with periodic boundary condiitons.
    grid = ArakawaCGrid(xbounds, nx, [-0.5*xbounds[1], 0.5*xbounds[1]], periodicX=True)
    solver.model.grid = grid
    
    # Set blob initial condition.
    mu = np.array([0.2*xbounds[1], 0])
    var = ((5*dx)**2*np.array([2, 2])**2)
    solver.model.setBlobInitialCondition(mu, var, 100)
    
    # Run the solver and store every time step.
    solver.store = True
    solver.run()
        
    # Create animation of surface elavation (eta).
    animateSurfaceElevation(solver, "kelvinWaveEta.gif")
    
    # Create animation of velocity field.
    animateVelocityField(solver, 3, "kelvinWaveVelocity.gif")
    

if __name__ == "__main__":
    
    # Warning: these take a very long time to run.
    runAndAnimateGravityWaves()
    runAndAnimateKelvinWaves()
    