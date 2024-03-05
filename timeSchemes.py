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
    dx = 20e3
    nx = int((xbounds[1] - xbounds[0])/dx)
    grid = ArakawaCGrid(xbounds, nx)
    
    # Time stepping information.
    dt = 100
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
    ## HEIGHT PERTURBATION STEP ##
        
    # Departure point for height perturbation (will need to think about bcs).
    Xetadp = Xmid - Umid*dt
    Yetadp = Ymid - Vmid*dt
        
    # Interpolate height perturbation field.
    interpH = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), grid.hField, 
                                      bounds_error=False, fill_value=None)
    hField2 = interpH((Yetadp, Xetadp))
            
    # # Calculate dudx and dvdy at departure point.
    # interpU = RegularGridInterpolator((), grid.uField, 
    #                                   bounds_error=False, fill_value=None)
    # dudx_dp = None
    
    # inperpV = RegularGridInterpolator((), grid.vField, 
    #                                   bounds_error=False, fill_value=None)
    # dvdy_dp = None
    
    # Find new eta.
    solver.model.grid.hField = hField2 + dt*model.eqns[0](solver.model.grid) # + departure point.
    
    # Update solver grid (D':).
    solver.model.grid.fields["eta"] = solver.model.grid.hField
    
    #%%
    ## U-VELOCITY STEP ## 
    
    # Departure point for internal u-velocity (will need to think about bcs).
    Xudp = X[:-1, 1:-1] - grid.uField[:, 1:-1]*dt
    Yudp = Ymid[:, :-1] - grid.vOnUField()*dt
        
    # Interpolate u-velocity field.
    interpU = RegularGridInterpolator((Ymid[:, 0], X[0, 1:-1]), grid.uField[:, 1:-1], 
                                      bounds_error=False, fill_value=None)
    uField2 = interpU((Yudp, Xudp))
    
    # Find new u.
    solver.model.grid.uField[:, 1:-1] = uField2 + dt*model.eqns[1](solver.model.grid) # + departure points
    
    # Update the fields viewer.
    solver.model.grid.fields["uVelocity"] = solver.model.grid.uField[:, 1:-1]
    
    #%%
    ## V-VELOCITY STEP ##
    
    # Departure point for internal v-velocity (will need to think about bcs).
    Xvdp = Xmid[:-1, :] - grid.uOnVField()*dt
    Yvdp = Y[1:-1, :-1] - grid.vField[1:-1, :]*dt
    
    # Interpolate v-velocity field.
    interpV = RegularGridInterpolator((Y[1:-1, 0], Xmid[0, :]), grid.vField[1:-1, :], 
                                      bounds_error=False, fill_value=None)
    vField2 = interpV((Yvdp, Xvdp))
    
    # Find new v.
    solver.model.grid.vField[1:-1, :] = vField2 + dt*model.eqns[2](solver.model.grid) # + departure points
    
    # Update the fields viewer.
    solver.model.grid.fields["vVelocity"] = solver.model.grid.vField[1:-1, :]
    
    plotContourSubplot(solver.model.grid)
    