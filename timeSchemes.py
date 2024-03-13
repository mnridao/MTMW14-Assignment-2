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

# OLD - From assignment 1
def RK4SchemeCoupled(funcs, grid, dt, nt):
    """ 
    """
    
    # hmm
    k1 = funcs[0](grid)
    l1 = funcs[1](grid)
    m1 = funcs[2](grid)
    
    ## SECOND STAGE ##
    gridP = grid.copy()
    
    gridP.fields["eta"] += 0.5*dt*k1 
    gridP.fields["uVelocity"] += 0.5*dt*l1
    gridP.fields["vVelocity"] += 0.5*dt*m1 
    
    k2 = funcs[0](gridP)
    l2 = funcs[1](gridP)
    m2 = funcs[2](gridP)
    
    ## THIRD STAGE ##
    gridP = grid.copy()
    
    gridP.fields["eta"] += 0.5*dt*k2 
    gridP.fields["uVelocity"] += 0.5*dt*l2
    gridP.fields["vVelocity"] += 0.5*dt*m2
    
    k3 = funcs[0](gridP)
    l3 = funcs[1](gridP)
    m3 = funcs[2](gridP)
    
    ## FOURTH STAGE ##
    gridP = grid.copy()
    
    gridP.fields["eta"] += dt*k3 
    gridP.fields["uVelocity"] += dt*l3 
    gridP.fields["vVelocity"] += dt*m3 
    
    k4 = funcs[0](gridP)
    l4 = funcs[1](gridP)
    m4 = funcs[2](gridP)
    
    ## Runge-Kutte step.
    gridP.fields["eta"] += dt*(k1 + 2*k2 + 2*k3 + k4)/6
    grid.fields["uVelocity"] += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    grid.fields["vVelocity"] += dt*(m1 + 2*m2 + 2*m3 + m4)/6


# def RK4SchemeCoupled(funcs, dt, nt, *phi0s):
#     """ 
#     Runge-Kutta 4 scheme for coupled equations.
    
#     Inputs
#     -------
    
#     Returns
#     -------
#     """
    
#     # Initialise array of k values (1 - 4 for RK4).
#     k = np.zeros(shape=(4, len(funcs)))
    
#     phiPs = phi0s
#     nti = nt - 1
#     for i in range(4):
#         for j, func in enumerate(funcs):
            
#             # Update k value.
#             k[i, j] = func(nti*dt, dt, *phiPs)
            
#         # Update RK prediction value for RK4.
#         phiPs = phi0s + dt * k[i, :] * (0.5 if i < 2 else 1)
#         nti = nt - 0.5 if i < 2 else 0
        
#     # Calculate new timestep.
#     phi = np.zeros(shape=len(funcs))
#     for i, (func, phi0) in enumerate(zip(funcs, phi0s)):
        
#         # Runge-Kutta step.
#         phi[i] = phi0 + dt * (k[0, i] + 2*k[1, i] + 2*k[2, i] + k[3, i]) / 6
    
#     return phi

# TODO: RK4
# TODO: Finish semi lagrangian
# TODO: Lax-wendroff?
# TODO: Semi implicit semi lagrangian?

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
    # solver.run()
    
    params = Parameters()
    
    interpMethod = "cubic"
    
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
                                      bounds_error=False, fill_value=None, method=interpMethod)
    interpV = RegularGridInterpolator((Y[1:-1, 0], Xmid[0, :]), grid.vField[1:-1, :], 
                                      bounds_error=False, fill_value=None, method=interpMethod)
    interpH = RegularGridInterpolator((Ymid[:, 0], Xmid[0, :]), grid.hField, 
                                      bounds_error=False, fill_value=None, method=interpMethod)
    #%%
    
    for _ in range(705):
    
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
                                          bounds_error=False, fill_value=None, method=interpMethod)
        
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
                                          bounds_error=False, fill_value=None, method=interpMethod)
        
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
                                          bounds_error=False, fill_value=None, method=interpMethod)
        
    plotContourSubplot(solver.model.grid)