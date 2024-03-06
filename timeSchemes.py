"""
MTMW14 Assignment 2

Student ID: 31827379
"""

def forwardBackwardSchemeCoupled(funcs, grid, dt, nt):
    """ 
    
    Inputs
    -------
    
    Returns
    -------
    """
    # Forward-euler for scalar field (height perturbation).
    grid.fields[funcs[0].name] += dt*funcs[0](grid)
    
    # Iterate over velocity fields.
    for func in reversed(funcs[1:]) if nt%2 != 0 else funcs[1:]:
        
        # Forward-euler step for the current velocity field.
        grid.fields[func.name] += dt*func(grid)
    
    return

if __name__ == "__main__":

    import numpy as np
    
    from grids import ArakawaCGrid
    from equations import Parameters
    
    from solver import Solver, Model
    from equations import UVelocity, VVelocity, Eta
    
    from plotters import plotContourSubplot
    
    # Grid creation.
    xbounds = [0, 1e6]
    xL = xbounds[1]
    dx = 50e3
    nx = int((xbounds[1] - xbounds[0])/dx)
    grid = ArakawaCGrid(xbounds, nx)
    
    # Time stepping information.
    dt = 350
    endtime = 30*24*60**2 
    nt = int(np.ceil(endtime/dt))
    
    dy = dx
    yL = xL
    ny = nx
    
    scheme = forwardBackwardSchemeCoupled
    model = Model([Eta(), UVelocity(), VVelocity()], grid)
    solver = Solver(model, scheme, dt, 1)
    solver.run()
    
    params = Parameters()
    
    #%% One time step later.
    from scipy.interpolate import RegularGridInterpolator
    
    X = grid.X
    Y = grid.Y
    
    # Find eta grid (should probably have this in ArakawaCGrid?)
    midPointsX = np.linspace(0.5*dx, xL-0.5*dx, nx)
    midPointsY = np.linspace(0.5*dy, yL-0.5*dy, ny)
    
    # Mid point grid.
    Xmid, Ymid = np.meshgrid(midPointsX, midPointsY)
    
    # Velocity fields interpolated on eta field.    
    Umid = 0.5*(grid.uField[:, :-1] + grid.uField[:, 1:])
    Vmid = 0.5*(grid.vField[:-1, :] + grid.vField[1:, :])
    
    #%%
    
    ## Current time step interpolators.
    interpU = RegularGridInterpolator((Ymid[:, 0], X[0, 1:-1]), grid.uField[:, 1:-1],
                                      bounds_error=False, fill_value=None)
    interpV = RegularGridInterpolator((Y[1:-1, 0], Xmid[0, :]), grid.vField[1:-1, :], 
                                      bounds_error=False, fill_value=None)
    interpH = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), grid.hField, 
                                      bounds_error=False, fill_value=None)
    #%%
    
    for _ in range(nt):
    
        ## HEIGHT PERTURBATION STEP ##
            
        # -- Half time step -- #
        # grid = ArakawaCGrid(xbounds, nx) # Make a copy of the current time step.
        
        # Departure point for height perturbation (half time step).
        Xstar = Xmid - Umid*dt/2
        Ystar = Ymid - Vmid*dt/2
                
        # Find the velocity at half time step departure point.
        uStar = interpU((Ystar, Xstar))
        vStar = interpV((Ystar, Xstar))
        
        # -- Full time step -- #
        Xetadp = Xmid - uStar*dt
        Yetadp = Ymid - vStar*dt
        
        # Interpolate height perturbation field at half a time step.
        hField2 = interpH((Yetadp, Xetadp))
        
        # TODO: Think about how to interpolate forcings onto the departure point.
        
        
        
        # Find new eta.
        solver.model.grid.hField = hField2 + dt*model.eqns[0](solver.model.grid)
        
        # Update solver grid.
        solver.model.grid.fields["eta"] = solver.model.grid.hField
        
        # plotContourSubplot(solver.model.grid)
        
        # Update the eta interpolater.
        interpH = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), solver.model.grid.hField, 
                                          bounds_error=False, fill_value=None)
        
        ## U-VELOCITY STEP ## 
        
        # -- Half time step -- #
        Xstar = X[:-1, 1:-1] - grid.uField[:, 1:-1]*dt/2
        Ystar = Ymid[:, :-1] - grid.vOnUField()*dt/2
        
        # Find velocities at the half time step departure points.
        uStar = interpU((Ystar, Xstar))
        vStar = interpV((Ystar, Xstar))
        
        # Departure point for internal u-velocity (will need to think about bcs).
        Xudp = X[:-1, 1:-1] - uStar*dt
        Yudp = Ymid[:, :-1] - vStar*dt
            
        # Interpolate u-velocity field.
        uField2 = interpU((Yudp, Xudp))
        
        # Find new u.
        solver.model.grid.uField[:, 1:-1] = uField2 + dt*model.eqns[1](solver.model.grid) # + departure points
        
        # Update the fields viewer.
        solver.model.grid.fields["uVelocity"] = solver.model.grid.uField[:, 1:-1]
        
        # Update the u-velocity interpolator.
        interpU = RegularGridInterpolator((Ymid[:, 0], X[0, 1:-1]), solver.model.grid.uField[:, 1:-1],
                                          bounds_error=False, fill_value=None)
        
        # plotContourSubplot(solver.model.grid)
        
        ## V-VELOCITY STEP ##
        
        # -- Half time step -- #
        Xstar = Xmid[:-1, :] - grid.uOnVField()*dt/2
        Ystar = Y[1:-1, :-1] - grid.vField[1:-1, :]*dt/2
        
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
                                          bounds_error=False, fill_value=None)
        
    plotContourSubplot(solver.model.grid)