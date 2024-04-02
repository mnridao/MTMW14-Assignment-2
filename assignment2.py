"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt

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
    plotters.plotContourSubplot(uSol, vSol, etaSol)

#### GLOBAL GRID VARIABLES ####
xbounds = [0, 1e6]
dx = 25e3
nx = int((xbounds[1] - xbounds[0])/dx)
grid = ArakawaCGrid(xbounds, nx, periodicX=False)

#### GLOBAL TIME STEPPING VARIABLES ####
dt = 0.99*helpers.calculateTimestepCFL(100, dx)
endtime = 30*24*60**2 
nt = int(np.ceil(endtime/dt))

#### GLOBAL MODEL AND SOLVER ####
model  = Model([Eta(), UVelocity(), VVelocity()], grid)
scheme = helpers.setScheme("forwardBackward")
solver = Solver(model, scheme, dt, nt)

if __name__ == "__main__":
        
    # Run Task C.    
    runTaskC()
    
    #%% Run Task D.
    