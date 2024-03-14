"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

from grids import ArakawaCGrid
from equations import Parameters

from solver import Solver 
from model import Model
from equations import UVelocity, VVelocity, Eta

from plotters import plotContourSubplot

"""
Thoughts:
        
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
    
    Xmid = grid.Xmid 
    Ymid = grid.Ymid
        
    #%%
    
    ## Current time step interpolators.
    interpU = RegularGridInterpolator((Ymid[:, 0], X[0, 1:-1]), grid.uField[:, 1:-1],
                                      bounds_error=False, fill_value=None, method=interpMethod)
    interpV = RegularGridInterpolator((Y[1:-1, 0], Xmid[0, :]), grid.vField[1:-1, :], 
                                      bounds_error=False, fill_value=None, method=interpMethod)
    
    # interpU = RegularGridInterpolator((Ymid[:, 0], X[0, :]), grid.uField,
    #                                   bounds_error=False, fill_value=None, method=interpMethod)
    # interpV = RegularGridInterpolator((Y[:, 0], Xmid[0, :]), grid.vField, 
    #                                   bounds_error=False, fill_value=None, method=interpMethod)
    
    interpH = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), grid.hField, 
                                      bounds_error=False, fill_value=None, method=interpMethod)
    #%%
    
    for t in range(nt):
        
        # Calculate all departure points now (PLV email).
        
        ## ETA DEPARTURE POINT ##
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
        
        ## U VELOCITY ##
        
        # -- Half time step -- #
        Xstar = X[:-1, 1:-1] - solver.model.grid.uField[:, 1:-1]*dt/2
        Ystar = Ymid[:, :-1] - solver.model.grid.vOnUField()*dt/2
        
        # Find velocities at the half time step departure points.
        uStar = interpU((Ystar, Xstar))
        vStar = interpV((Ystar, Xstar))
        
        # Departure point for internal u-velocity (will need to think about bcs).
        Xudp = X[:-1, 1:-1] - uStar*dt
        Yudp = Ymid[:, :-1] - vStar*dt
        
        ## V VELOCITY ##
        
        # -- Half time step -- #
        Xstar = Xmid[:-1, :] - solver.model.grid.uOnVField()*dt/2
        Ystar = Y[1:-1, :-1] - solver.model.grid.vField[1:-1, :]*dt/2
        
        # Find velocities at the half time step departure point.
        uStar = interpU((Ystar, Xstar))
        vStar = interpV((Ystar, Xstar))
        
        # Departure point for internal v-velocity (will need to think about bcs).
        Xvdp = Xmid[:-1, :] - uStar*dt
        Yvdp = Y[1:-1, :-1] - vStar*dt
        
        #######################################################################
        
        ## HEIGHT PERTURBATION STEP ##
            
        # # -- Half time step -- #
        
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
        
        # Interpolate height perturbation field at half a time step.
        hField2 = interpH((Yetadp, Xetadp))
                
        dudx = grid.dudxField()
        dvdy = grid.dvdyField()
        
        # Find which cell the departure point is at.
        
        dudxInterp = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), dudx, 
                                             bounds_error=False, fill_value=None, method=interpMethod)
        dvdyInterp = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), dvdy, 
                                             bounds_error=False, fill_value=None, method=interpMethod)
        
        # Ew
        dudxDP = dudxInterp((Yetadp, Xetadp))
        dvdyDP = dvdyInterp((Yetadp, Xetadp))
        
        # Find new eta.
        solver.model.grid.fields["eta"] += 0.5*dt*(model.eqns[0](solver.model.grid) +
                                                   model.eqns[0].forcings(dudxDP, dvdyDP))
        
        # Update solver grid.
        # solver.model.grid.fields["eta"] = solver.model.grid.hField
        
        # plotContourSubplot(solver.model.grid)
        
        # Update the eta interpolater.
        interpH = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), solver.model.grid.hField, 
                                          bounds_error=False, fill_value=None, method=interpMethod)    
        #%%
        ## U-VELOCITY STEP ## 
        
        # # -- Half time step -- #
        # Xstar = X[:-1, 1:-1] - 0.5*dt*solver.model.grid.uField[:, 1:-1]
        # Ystar = Ymid[:, :-1] - 0.5*dt*solver.model.grid.vOnUField()
                
        # # Find velocities at the half time step departure points.
        # uStar = interpU((Ystar, Xstar))
        # vStar = interpV((Ystar, Xstar))
        
        # # Departure point for internal u-velocity (will need to think about bcs).
        # Xudp = X[:-1, 1:-1] - uStar*dt
        # Yudp = Ymid[:, :-1] - vStar*dt
            
        # u-velocity field at the departure point.
        uField2 = interpU((Yudp, Xudp))
        
        # detadx interpolator.
        detadx = grid.detadxField()
        
        # detadx interpolator.
        detadxInterp = RegularGridInterpolator((Ymid[:, 0], X[0, 1:-1]), detadx, 
                                             bounds_error=False, fill_value=None, method=interpMethod)
        detadxDP = detadxInterp((Yudp, Xudp))
        
        # Find new u.
        uDP = uField2
        vDP = interpV((Yudp, Xudp)) # Don't need to interpolate on u, already there
        
        solver.model.grid.fields["uVelocity"] += (0.5*dt*
                                                  (model.eqns[1](solver.model.grid) + 
                                                    model.eqns[1].forcings(uDP, vDP, detadxDP, Yudp, xL)))
                                
        # Update the u-velocity interpolator.
        interpU = RegularGridInterpolator((Ymid[:, 0], X[0, 1:-1]), grid.uField[:, 1:-1],
                                          bounds_error=False, fill_value=None, method=interpMethod)
        
        # plotContourSubplot(solver.model.grid)
        
        #%%
        ## V-VELOCITY STEP ##
        
        # # -- Half time step -- #
        # Xstar = Xmid[:-1, :] - solver.model.grid.uOnVField()*dt/2
        # Ystar = Y[1:-1, :-1] - solver.model.grid.vField[1:-1, :]*dt/2
        
        # # Find velocities at the half time step departure point.
        # uStar = interpU((Ystar, Xstar))
        # vStar = interpV((Ystar, Xstar))
        
        # # Departure point for internal v-velocity (will need to think about bcs).
        # Xvdp = Xmid[:-1, :] - uStar*dt
        # Yvdp = Y[1:-1, :-1] - vStar*dt
        
        vField2 = interpV((Yvdp, Xvdp))
        
        # detadx interpolator.
        detady = grid.detadyField()
        detadyInterp = RegularGridInterpolator((Y[1:-1, 0], Xmid[0, :]), detady, 
                                             bounds_error=False, fill_value=None, method=interpMethod)
        detadyDP = detadyInterp((Yvdp, Xvdp))
        
        # Find departure point forcings.
        v = vField2 
        u = interpU((Yvdp, Xvdp))
        
        # Find new v.
        solver.model.grid.fields["vVelocity"] += (0.5*dt*
                                                  (model.eqns[2](solver.model.grid) + 
                                                    model.eqns[2].forcings(u, v, detadyDP, Yvdp)))
        
        # solver.model.grid.fields["vVelocity"] += dt*model.eqns[2](solver.model.grid)
        
        # Update the interpolator.
        interpV = RegularGridInterpolator((Y[1:-1, 0], Xmid[0, :]), solver.model.grid.vField[1:-1, :], 
                                          bounds_error=False, fill_value=None, method=interpMethod)
        
        plotContourSubplot(solver.model.grid)