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
    
    # Grid creation.
    xbounds = [0, 2e7]
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
    
    params = Parameters()
    
    #%% One time step later.
    from scipy.interpolate import RegularGridInterpolator
    
    X = grid.X
    Y = grid.Y
    
    # Find eta grid (should probably have this in ArakawaCGrid?)
    midPointsX = np.linspace(0.5*dx, xL-0.5*dx, nx)
    midPointsY = np.linspace(0.5*dy, yL-0.5*dy, ny)
    
    Xmid, Ymid = np.meshgrid(midPointsX, midPointsY)
    
    # Velocity fields interpolated on eta field.    
    Umid = 0.5*(grid.uField[:, :-1] + grid.uField[:, 1:])
    Vmid = 0.5*(grid.vField[:-1, :] + grid.vField[1:, :])
    
    ## HEIGHT PERTURBATION STEP ##
    
    # Departure point for height perturbation (will need to think about bcs).
    Xetadp = Xmid - Umid*dt
    Yetadp = Ymid - Vmid*dt
        
    # Interpolate height perturbation field.
    interpH = RegularGridInterpolator((Xmid[0, :], Ymid[:, 0]), grid.hField)
    hField2 = interpH((Xetadp, Yetadp))
    
    # Calculate dudx and dvdy.
    dudx = (grid.uField[:, 1:] - grid.uField[:, :-1]) / grid.dx
    dvdy = (grid.vField[1:, :] - grid.vField[:-1, :]) / grid.dy
    
    # Find new eta.
    hField = hField2 + dt*( - params.H*(dudx + dvdy))
    
    ## U-VELOCITY STEP ## 
    
    # Departure point for internal u-velocity (will need to think about bcs).
    Xudp = X[:-1, 1:-1] - grid.uField[:, 1:-1]*dt
    Yudp = Ymid[:, :-1] - grid.vOnUField()*dt
        
    # Interpolate u-velocity field.
    # interpU = RegularGridInterpolator((X[0, 1:-1], Ymid[1:, 0]), grid.uField[:, 1:-1])
    
    # Find new u.
    
    
    ## V-VELOCITY STEP ##
    
    # Departure point for internal v-velocity (will need to think about bcs).
    Xvdp = Xmid[:-1, :] - grid.uOnVField()*dt
    Yvdp = Y[:, :-1] - grid.vField*dt
    
    # Interpolate v-velocity field.
    # interpV = RegularGridInterpolator(())
    
    # Find new v.
    
    #%%
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np
    def f(x, y, z):
        return 2 * x**3 + 3 * y**2 - z
    x = np.linspace(1, 4, 11)
    y = np.linspace(4, 7, 22)
    z = np.linspace(7, 9, 33)
    xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    data = f(xg, yg, zg)