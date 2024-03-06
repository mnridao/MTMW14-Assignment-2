"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

from grids import ArakawaCGrid
from equations import Parameters

from solver import Solver, Model
from equations import UVelocity, VVelocity, Eta

from plotters import plotContourSubplot

"""
Thoughts:
    
    - Bring back the State class, just pass in fields and points to eqns?
    - Have gradient fields in the grid class???
    
    - rethink how information is passed into equations
    - want semi lagrangian to look nice
    
    - if using forcings method for equations make all calls using this function in sl?
    - might just go with this?
"""


if __name__ == "__main__":
    
    # Grid creation.
    xbounds = [0, 1e6]
    xL = xbounds[1]
    dx = 50e3
    nx = int((xbounds[1] - xbounds[0])/dx)
    grid = ArakawaCGrid(xbounds, nx)
    
    # Time stepping information.
    dt = 350
    endtime = 100*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    dy = dx
    yL = xL
    ny = nx
    
    scheme = None
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    solver = Solver(model, scheme, dt, 1)
    # solver.run()
    
    params = Parameters()
    
    interpMethod = "linear"
    
    #%% One time step later.
    from scipy.interpolate import RegularGridInterpolator
    
    X = grid.X
    Y = grid.Y
    
    # Find eta grid (should probably have this in ArakawaCGrid?)
    midPointsX = np.linspace(0.5*dx, xL-0.5*dx, nx)
    midPointsY = np.linspace(0.5*dy, yL-0.5*dy, ny)
    
    # Mid point grid.
    Xmid, Ymid = np.meshgrid(midPointsX, midPointsY)
    
    # # Velocity fields interpolated on eta field.    
    # Umid = 0.5*(grid.uField[:, :-1] + grid.uField[:, 1:])
    # Vmid = 0.5*(grid.vField[:-1, :] + grid.vField[1:, :])
    
    #%%
    
    ## Current time step interpolators.
    interpU = RegularGridInterpolator((Ymid[:, 0], X[0, 1:-1]), grid.uField[:, 1:-1],
                                      bounds_error=False, fill_value=None, method=interpMethod)
    interpV = RegularGridInterpolator((Y[1:-1, 0], Xmid[0, :]), grid.vField[1:-1, :], 
                                      bounds_error=False, fill_value=None, method=interpMethod)
    interpH = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), grid.hField, 
                                      bounds_error=False, fill_value=None, method=interpMethod)
    #%%
    
    for t in range(nt):
        
        ## Calculate all departure points now (PLV email).
        
        # ## ETA DEPARTURE POINT ##
        # # Update mid point velocities.
        # Umid = 0.5*(solver.model.grid.uField[:, :-1] + solver.model.grid.uField[:, 1:])
        # Vmid = 0.5*(solver.model.grid.vField[:-1, :] + solver.model.grid.vField[1:, :])

        # # Departure point for height perturbation (half time step).
        # Xstar = Xmid - 0.5*dt*Umid
        # Ystar = Ymid - 0.5*dt*Vmid
                
        # # Find the velocity at half time step departure point.
        # uStar = interpU((Ystar, Xstar))
        # vStar = interpV((Ystar, Xstar))
        
        # # -- Full time step -- #
        # Xetadp = Xmid - dt*uStar
        # Yetadp = Ymid - dt*vStar
        
        # ## U VELOCITY ##
        
        # # -- Half time step -- #
        # Xstar = X[:-1, 1:-1] - solver.model.grid.uField[:, 1:-1]*dt/2
        # Ystar = Ymid[:, :-1] - solver.model.grid.vOnUField()*dt/2
        
        # # Find velocities at the half time step departure points.
        # uStar = interpU((Ystar, Xstar))
        # vStar = interpV((Ystar, Xstar))
        
        # # Departure point for internal u-velocity (will need to think about bcs).
        # Xudp = X[:-1, 1:-1] - uStar*dt
        # Yudp = Ymid[:, :-1] - vStar*dt
        
        # ## V VELOCITY ##
        
        # # -- Half time step -- #
        # Xstar = Xmid[:-1, :] - solver.model.grid.uOnVField()*dt/2
        # Ystar = Y[1:-1, :-1] - solver.model.grid.vField[1:-1, :]*dt/2
        
        # # Find velocities at the half time step departure point.
        # uStar = interpU((Ystar, Xstar))
        # vStar = interpV((Ystar, Xstar))
        
        # # Departure point for internal v-velocity (will need to think about bcs).
        # Xvdp = Xmid[:-1, :] - uStar*dt
        # Yvdp = Y[1:-1, :-1] - vStar*dt
        
        #######################################################################
        
        ## HEIGHT PERTURBATION STEP ##
            
        # -- Half time step -- #
        
        # Update mid point velocities.
        Umid = 0.5*(solver.model.grid.uField[:, :-1] + solver.model.grid.uField[:, 1:])
        Vmid = 0.5*(solver.model.grid.vField[:-1, :] + solver.model.grid.vField[1:, :])

        # Departure point for height perturbation (half time step).
        Xstar = Xmid - 0.5*dt*Umid
        Ystar = Ymid - 0.5*dt*Vmid
                
        # Find the velocity at half time step departure point.
        uStar = interpU((Ystar, Xstar))
        vStar = interpV((Ystar, Xstar))
        
        # -- Full time step -- #
        Xetadp = Xmid - dt*uStar
        Yetadp = Ymid - dt*vStar
        
        # Interpolate height perturbation field at half a time step.
        hField2 = interpH((Yetadp, Xetadp))
        
        # TODO: Think about how to interpolate forcings onto the departure point.
        dudx = (grid.uField[:, 1:] - grid.uField[:, :-1]) / grid.dx
        dvdy = (grid.vField[1:, :] - grid.vField[:-1, :]) / grid.dy
        
        # Find which cell the departure point is at.
        
        dudxInterp = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), dudx, 
                                             bounds_error=False, fill_value=None, method=interpMethod)
        dvdyInterp = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), dvdy, 
                                             bounds_error=False, fill_value=None, method=interpMethod)
        
        # Ew
        dudxDP = dudxInterp((Yetadp, Xetadp))
        dvdyDP = dvdyInterp((Yetadp, Xetadp))
        
        # Find new eta.
        solver.model.grid.hField = hField2 + 0.5*dt*(model.eqns[0](solver.model.grid) +
                                                     model.eqns[0].forcings(dudxDP, dvdyDP)) # Don't like this.
        
        # Update solver grid.
        solver.model.grid.fields["eta"] = solver.model.grid.hField
        
        # plotContourSubplot(solver.model.grid)
        
        # Update the eta interpolater.
        interpH = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), solver.model.grid.hField, 
                                          bounds_error=False, fill_value=None, method=interpMethod)    
        #%%
        ## U-VELOCITY STEP ## 
        
        # -- Half time step -- #
        Xstar = X[:-1, 1:-1] - solver.model.grid.uField[:, 1:-1]*dt/2
        Ystar = Ymid[:, :-1] - solver.model.grid.vOnUField()*dt/2
        
        # Find velocities at the half time step departure points.
        uStar = interpU((Ystar, Xstar))
        vStar = interpV((Ystar, Xstar))
        
        # Departure point for internal u-velocity (will need to think about bcs).
        Xudp = X[:-1, 1:-1] - uStar*dt
        Yudp = Ymid[:, :-1] - vStar*dt
            
        # u-velocity field at the departure point.
        uField2 = interpU((Yudp, Xudp))
        
        # Find new u.
        solver.model.grid.uField[:, 1:-1] = uField2 + dt*model.eqns[1](solver.model.grid) # + departure points
        
        # Update the fields viewer.
        solver.model.grid.fields["uVelocity"] = solver.model.grid.uField[:, 1:-1]
        
        # Update the u-velocity interpolator.
        interpU = RegularGridInterpolator((Ymid[:, 0], X[0, 1:-1]), solver.model.grid.uField[:, 1:-1],
                                          bounds_error=False, fill_value=None, method=interpMethod)
        
        # plotContourSubplot(solver.model.grid)
        
        ## V-VELOCITY STEP ##
        
        # -- Half time step -- #
        Xstar = Xmid[:-1, :] - solver.model.grid.uOnVField()*dt/2
        Ystar = Y[1:-1, :-1] - solver.model.grid.vField[1:-1, :]*dt/2
        
        # Find velocities at the half time step departure point.
        uStar = interpU((Ystar, Xstar))
        vStar = interpV((Ystar, Xstar))
        
        # Departure point for internal v-velocity (will need to think about bcs).
        Xvdp = Xmid[:-1, :] - uStar*dt
        Yvdp = Y[1:-1, :-1] - vStar*dt
        
        vField2 = interpV((Yvdp, Xvdp))
        
        # Find new v.
        solver.model.grid.vField[1:-1, :] = vField2 + dt*model.eqns[2](solver.model.grid) # + departure points
        
        # Update the fields viewer.
        solver.model.grid.fields["vVelocity"] = solver.model.grid.vField[1:-1, :]
        
        # Update the interpolator.
        interpV = RegularGridInterpolator((Y[1:-1, 0], Xmid[0, :]), solver.model.grid.vField[1:-1, :], 
                                          bounds_error=False, fill_value=None, method=interpMethod)
        
    plotContourSubplot(solver.model.grid)