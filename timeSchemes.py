"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def forwardBackwardSchemeCoupled(funcs, grid, dt, nt):
    """ 
    Forward-backward time scheme for coupled equations. The eta equation is 
    solved first, then each iteration the order that the u- and v-velocities 
    are solved is switched. Each equation solved using forward euler.
    
    Inputs
    -------
    funcs : list of callable objects
            RHS of differenetial equation to be solved numerically.
    grid  : ArakawaCGrid object
            Object containing the domain and state information for the problem.
            Passed by reference.
    dt    : float
            Timestep.
    nt    : int 
            Current iteration of the simulation.
    
    Returns
    -------
    None
    """
    # Forward-euler for scalar field (height perturbation).
    grid.fields[funcs[0].name] += dt*funcs[0](grid)
    
    # Iterate over velocity fields.
    for func in reversed(funcs[1:]) if nt%2 != 0 else funcs[1:]:
        
        # Forward-euler step for the current velocity field.
        grid.fields[func.name] += dt*func(grid)

def RK4SchemeCoupled(funcs, grid, dt, nt):
    """ 
    Runge-Kutte-4 scheme for coupled equations.
    
    Inputs
    -------
    funcs : list of callable objects
            RHS of differenetial equation to be solved numerically.
    grid  : ArakawaCGrid object
            Object containing the domain and state information for the problem.
            Passed by reference.
    dt    : float
            Timestep.
    nt    : int 
            Current iteration of the simulation.
    
    Returns
    -------
    None
    """
    
    # Initialise k as dict (please think of something better).
    kfunc = {}
    for func in funcs:
        kfunc[func.name] = np.zeros(shape=(4, *grid.fields[func.name].shape))
            
    gridP = grid
    for i in range(4):
        for func in funcs:
                        
            # Find k_i for the current function.
            kfunc[func.name][i] = func(gridP)
        
        if i==3: break # break out early to avoid mess below.
        
        # Make a copy to reset fields to previous timestep.
        gridP = grid.copy()  #oof
        
        # Update the intermediate prediction grid (another loop!).
        for func in funcs:
                        
            # Update the prediction grid for the next prediction.
            gridP.fields[func.name] += dt*kfunc[func.name][i]*(0.5 if i < 2 else 1)
            
    # Runge-kutte step (loop number 4!).
    for func in funcs:
        
        k = kfunc[func.name]
        grid.fields[func.name] += dt*(k[0] + 2*k[1] + 2*k[2] + k[3])/6

class SemiImplicitSchemeCoupled:
    """
    Semi-Implicit scheme for coupled equations, treating the gravity terms as
    implicit and the remaining forcing terms as explicit.
    """
    
    def __init__(self, model, dt):
        """
        Initialisation of the object. Arguments passed are necessary for the 
        creation of the implicit matrix inverse.
        
        Inputs
        -------
        model : Model object 
                Object containing the problem and the grid.
        dt    : float
                Timestep.
        """
        
        # Default implicit matrix inverse.
        self.Linv = None
        
        # Create the implicit matrix that will be used each time step.
        self.createImplicitMatrix(model.grid, model.eqns[0].params, dt)
        
    def __call__(self, funcs, grid, dt, nt):
        """ 
        Semi-implicit scheme for coupled equations. The discretisation scheme 
        used is forward euler, and the gravity terms are treated fully 
        implicitly, i.e. no mixed explicit-implicit treatment of gravity waves
        like the trapezoidal method in the notes.
        
        This is very messy - I would have liked to clean it up a bit but I ran 
        out of time. 
        
        Inputs
        -------
        funcs : list of callable objects
                RHS of differenetial equation to be solved numerically.
        grid  : ArakawaCGrid object
                Object containing the domain and state information for the 
                problem.
                Passed by reference.
        dt    : float
                Timestep.
        nt    : int 
                Current iteration of the simulation.
        
        Returns
        -------
        None
        """
        
        # Calculate A on u-grid.
        A = self.calculateAgrid(grid, funcs[1], dt)
        
        # Calculate B on v-grid.
        B = self.calculateBgrid(grid, funcs[2], dt)
        
        # Calculate C on eta-grid.
        C = grid.hField.copy()
        
        # Calculate gradients of A and B (on eta grid).
        dAdx = grid.forwardGradientFieldX(A)
        dBdy = grid.forwardGradientFieldY(B)
        
        # Calculate the (flattened) explicit forcings matrix.
        F = (C - dt*funcs[0].params.H*(dAdx + dBdy)).flatten()
        
        # Update the eta field by inverting + solving matrix equation.
        grid.fields["eta"][:] = np.matmul(self.Linv, F).reshape(grid.hField.shape)
        
        # Update the velocity fields - [:] needed to not lose connection.
        grid.fields["uVelocity"][:] = ((A if grid.periodicX else A[:, 1:-1]) - 
                                       dt*funcs[0].params.g * grid.detadxField())
        grid.fields["vVelocity"][:] = (B[1:-1, :] - dt*funcs[0].params.g * 
                                       grid.detadyField())
            
    def calculateAgrid(self, grid, func, dt):
        """ 
        Calculates the A grid used in the calculation of the explicit forcings
        array. I would have liked to make these functions more general.
        
        Inputs
        -------
        grid  : ArakawaCGrid object
                Object containing the domain and state information for the 
                problem.
        func  : UVelocity object.
                RHS of differenetial u-velocity equation to be solved numerically.
        dt    : float
                Timestep.
        
        Returns
        -------
        A : np array
        """
        # Initialise as copy of u-velocity array. 
        A = grid.uField.copy()
        
        # Account for different indexing from different boundary conditions.
        if grid.periodicX:
            A += dt*(func.explicitTerms(grid))
        else:
            A[:, 1:-1] += dt*(func.explicitTerms(grid))
        
        return A
        
    def calculateBgrid(self, grid, func, dt):
        """ 
        Calculates the B grid used in the calculation of the explicit forcings
        array. I would have liked to make these functions more general.
        
        Inputs
        -------
        grid  : ArakawaCGrid object
                Object containing the domain and state information for the 
                problem.
        func  : VVelocity object.
                RHS of differenetial v-velocity equation to be solved numerically.
        dt    : float
                Timestep.
        
        Returns
        -------
        B : np array
        """
        # Initialise as copy of v-velocity array. 
        B = grid.vField.copy()
        
        # Account for different indexing from different boundary conditions.
        if grid.periodicY:
            B += dt*(func.explicitTerms(grid))
        else:
            B[1:-1, :] += dt*(func.explicitTerms(grid))
            
        return B      
        
    def createImplicitMatrix(self, grid, params, dt):
        """ 
        Creates the implicit matrix and inverts it. This is done when the 
        object is first initialised, and the inverted matrix is stored to avoid
        recalculating it at each iteration.
        
        Inputs
        -------
        grid  : ArakawaCGrid object
                Object containing the domain and state information for the 
                problem.
        params: Parameters object 
                Object containing the parameters for the problem.
        dt    : float
                Timestep.
        
        Returns
        -------
        None
        """
        
        # Create terms for L matrix diagonal.
        sx = params.g*params.H * (dt/grid.dx)**2
        sy = params.g*params.H * (dt/grid.dy)**2
            
        # Represent the terms right and left of the current point (ij)
        offDiagXTerms = [-sx]*(grid.nx - 1) + [0.]
        offDiagXTerms *= grid.nx
        
        # Represent the terms on rows above and below the current point (ij).
        offDiagYTerms = [-sy]*grid.ny*(grid.ny - 1)
            
        # Add off-diagonal elements to L.
        L = (np.diag(offDiagXTerms[:-1], k= 1) + 
             np.diag(offDiagXTerms[:-1], k=-1) + 
             np.diag(offDiagYTerms, k= grid.nx) + 
             np.diag(offDiagYTerms, k=-grid.nx))
                
        # Account for periodic boundary conditions.
        if grid.periodicX:
            periodicXTerms = [-sx] + [0.]*(grid.nx-1)
            periodicXTerms *= (grid.nx-1)
            periodicXTerms += [-sx]
            
            L += np.diag(periodicXTerms, k= grid.nx-1)
            L += np.diag(periodicXTerms, k=-grid.nx+1)
            
        # Add diagonal elements to L.
        L += np.diag((1 - np.sum(L, axis=1)))
        
        # Invert the matrix.
        self.Linv = np.linalg.inv(L)
    
class SemiLagrangianSchemeCoupled:
    """ 
    Semi-lagrangian scheme for coupled equations.
    
    Messy :(
    """
    
    def __init__(self, interpMethod="linear"):
        """ 
        Inputs
        -------
        interpMethod : string
            Gives a key for the interpolation method used by the interpolator. 
            Default is linear, and higher order interpolation methods come with 
            time and computational costs.
        """
        self.interpMethod = interpMethod
                
        # Store previous time step wind (for calculation of departure point).
        self.uFieldOld = None 
        self.vFieldOld = None
        
    def __call__(self, funcs, grid, dt, nt):
        """ 
        Semi-lagrangian scheme for coupled equations.
        
        This is very messy - I would have liked to clean it up a bit but I ran 
        out of time. 
        
        Inputs
        -------
        funcs : list of callable objects
                RHS of differenetial equation to be solved numerically.
        grid  : ArakawaCGrid object
                Object containing the domain and state information for the 
                problem.
                Passed by reference.
        dt    : float
                Timestep.
        nt    : int 
                Current iteration of the simulation.
        
        Returns
        -------
        None
        """           
        # Keep a copy of the original grid (won't be updated).
        gridOld = grid.copy()
                        
        # Update the height perturbation field first.
        self.updateField(grid, gridOld, funcs[0], dt)
        
        # Iterate over velocity fields.
        for func in reversed(funcs[1:]) if nt%2 != 0 else funcs[1:]:
            
            # Find departure point.
            self.updateField(grid, gridOld, func, dt)
        
        # Store previous time step for next round (used in dp calc).
        self.uFieldOld = gridOld.fields["uVelocity"]
        self.vFieldOld = gridOld.fields["vVelocity"]
            
    def updateField(self, grid, gridOld, func, dt):
        """
        Updates the current field (specified by func) based on the 
        semi-lagrangian method. Would have liked to clean this up but ran 
        out of time.
        
        Inputs
        -------
        grid    : ArakawaCGrid object
                  Object containing domain and state information - this grid 
                  object is updated after each equation variable is updated.
        gridOld : ArakawaCGrid object
                  Object containing domain and state information - this grid 
                  object is a copy of the original grid that is not updated.
        func    : BaseEqnSWE object 
                  RHS of differenetial equation to be solved numerically.
        dt      : float
                  Timestep.
        
        Returns
        -------
        None
        """
        
        # Calculate departure point.
        Xdp, Ydp = self.calculateDeparturePoint(gridOld, dt, func.name)
                
        # Departure point forcing calculations should be done at gridOld but blows up.
        # gridDP = gridOld
        gridDP = grid
        if func.name == "eta":
            
            fieldDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Ymid[:, 0], 
                                       gridDP.fields[func.name])
            forcingsDP = self.calculateEtaForcings(Xdp, Ydp, func, gridDP)
            
        elif func.name == "uVelocity":
            
            fieldDP = self.interpolate(Xdp, Ydp, grid.X[0, 1:-1], grid.Ymid[:, 0],
                                       gridDP.fields[func.name])
            forcingsDP = self.calculateUForcings(Xdp, Ydp, fieldDP, func, gridDP)

        elif func.name == "vVelocity":
            
            fieldDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Y[1:-1, 0], 
                                       gridDP.fields[func.name])  
            forcingsDP = self.calculateVForcings(Xdp, Ydp, fieldDP, func, gridDP)
        
        # Calculate forcings at current time step.
        forcingsNew = func(grid)
        
        # Update the current field ([:] required so that it doesn't break link).
        grid.fields[func.name][:] = (fieldDP + 0.5*dt*(forcingsNew + forcingsDP))
        
    def calculateDeparturePoint(self, gridOld, dt, funcName):
        """ 
        Calculates the departure point using a two-stage method.
        
        Inputs
        -------
        gridOld : ArakawaCGrid object
                  Object containing domain and state information - this grid 
                  object is a copy of the original grid that is not updated.
        dt      : float
                  Timestep.
        funcName: string
                  Name of the equation.
        
        Returns
        -------
        Xdp : np array
              X-departure point.
        Ydp : np array
              Y-departure point.
        """
        
        # Find the appropriate grid and fields for the current func.
        if funcName == "eta":
            
            # Eta stored at half cell points.
            # U = 0.5*(gridOld.uField[:, :-1] + gridOld.uField[:, 1:])
            # V = 0.5*(gridOld.vField[:-1, :] + gridOld.vField[1:, :])
            U = gridOld.uOnEtaField()
            V = gridOld.vOnEtaField()
            
            X = gridOld.Xmid 
            Y = gridOld.Ymid
            
        elif funcName == "uVelocity":
            
            # u-velocity stored at half cells in y and full cells in x.
            U = gridOld.uField[:, 1:-1]
            V = gridOld.vOnUField()
            
            X = gridOld.X[:-1, 1:-1]
            Y = gridOld.Ymid[:, :-1]
            
        elif funcName == "vVelocity":
            
            # v-velocity stored at half cells in x and full cells in y.
            U = gridOld.uOnVField()
            V = gridOld.vField[1:-1, :]
            
            X = gridOld.Xmid[:-1, :]
            Y = gridOld.Y[1:-1, :-1]
        
        # Intermediate departure point.
        Xstar = X - 0.5*dt*U
        Ystar = Y - 0.5*dt*V
                
        # Find velocities at intermediate departure point.
        Ustar = self.interpolate(Xstar, Ystar, gridOld.X[0, 1:-1], gridOld.Ymid[:, 0], 
                                 gridOld.uField[:, 1:-1])
        Vstar = self.interpolate(Xstar, Ystar, gridOld.Xmid[0, :], gridOld.Y[1:-1, 0], 
                                 gridOld.vField[1:-1, :])
        
        if np.any(self.uFieldOld):
            
            UstarOld = self.interpolate(Xstar, Ystar, gridOld.X[0, 1:-1], 
                                        gridOld.Ymid[:, 0], self.uFieldOld)
            VstarOld = self.interpolate(Xstar, Ystar, gridOld.Xmid[0, :], 
                                        gridOld.Y[1:-1, 0], self.vFieldOld)
            
            Ustar = 1.5*Ustar - 0.5*UstarOld
            Vstar = 1.5*Vstar - 0.5*VstarOld
            
        Xdp = X - dt*Ustar 
        Ydp = Y - dt*Vstar
                
        return Xdp, Ydp
    
    def calculateEtaForcings(self, Xdp, Ydp, func, grid):
        """ 
        Calculates the forcings for the eta equation.
        
        Inputs
        -------
        Xdp   : np array
                X-departure point coordinates.
        Ydp   : np array
                Y-departure point coordinates.
        func  : Eta object 
                Calculates RHS of eta differential equation.
        grid  : ArakawaCGrid object
                Contains domain and state information for the problem.
        
        Returns
        -------
        forcings : np array
            Forcing terms of the eta equation calculated at departure point.
        """
        
        # Forcing arguments for equation.
        dudxDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Ymid[:, 0], 
                                  grid.dudxField())
        dvdyDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Ymid[:, 0],
                                  grid.dvdyField())
        
        return func.forcings(dudxDP, dvdyDP)
        
    def calculateUForcings(self, Xdp, Ydp, fieldDP, func, grid):
        """ 
        Calculates the forcings for the u-velocity equation.
        
        Inputs
        -------
        Xdp   : np array
                X-departure point coordinates.
        Ydp   : np array
                Y-departure point coordinates.
        func  : UVelocity object 
                Calculates RHS of u-velocity differential equation.
        grid  : ArakawaCGrid object
                Contains domain and state information for the problem.
        
        Returns
        -------
        forcings : np array
            Forcing terms of the u-velocity equation calculated at departure point.
        """
        
        # Forcing arguments for equation.
        detadxDP = self.interpolate(Xdp, Ydp, grid.X[0, 1:-1], grid.Ymid[:, 0],
                                    grid.detadxField())
        vFieldDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Y[1:-1, 0],
                                    grid.fields["vVelocity"])


        return func.forcings(fieldDP, vFieldDP, detadxDP, grid.Ymid, grid.xbounds[1]) 
                
    def calculateVForcings(self, Xdp, Ydp, fieldDP, func, grid):
        """ 
        Calculates the forcings for the v-velocity equation.
        
        Inputs
        -------
        Xdp   : np array
                X-departure point coordinates.
        Ydp   : np array
                Y-departure point coordinates.
        func  : VVelocity object 
                Calculates RHS of v-velocity differential equation.
        grid  : ArakawaCGrid object
                Contains domain and state information for the problem.
        
        Returns
        -------
        forcings : np array
            Forcing terms of the v-velocity equation calculated at departure point.
        """
        
        # Forcing arguments for equations.
        detadyDP = self.interpolate(Xdp, Ydp, grid.Xmid[0, :], grid.Y[1:-1, 0], 
                                    grid.detadyField())
        uFieldDP = self.interpolate(Xdp, Ydp, grid.X[0, 1:-1], grid.Ymid[:, 0], 
                                    grid.fields["uVelocity"])
        
        return func.forcings(uFieldDP, fieldDP, detadyDP, grid.Y[1:-1, :-1])
    
    def interpolate(self, Xdp, Ydp, Xgrid, Ygrid, field):
        """ 
        Interpolate points from a grid. If points are outside the boundaries of 
        the given grid, values are extrapolated which is not exactly correct. 
        But has not been a massive problem, I think because velocities are small.
        
        Here I am using RegularGridInterpolator, and am constructing it each 
        time this function is called, instead of just storing it which might 
        be more efficient. I did this because I thought that I might create my 
        own interpolator - if I did this I would just need to change this 
        function and nothing else.
        
        Inputs
        -------
        Xdp   : np array
                X-departure point coordinates.
        Ydp   : np array
                Y-departure point coordinates.
        Xgrid : np array
                Array containing X coordinates of the interpolation grid.
        Ygrid : np array 
                Array containing Y coordinates of the interpolation grid.
        field : np array
                Array containing the field values that will be used for the 
                interpolation.
        
        Returns
        -------
        Interpolated points
        """
        
        # Construct the interpolator. 
        interp = RegularGridInterpolator((Ygrid, Xgrid), field, bounds_error=False, 
                                         fill_value=None, 
                                         method=self.interpMethod)
        
        # Return interpolated values at departure point.
        return interp((Ydp, Xdp))


class SemiImplicitSemiLagrangianCoupled(SemiLagrangianSchemeCoupled):
    """ 
    Ran out of time for SISL.
    """
    
    def __init__(self):
        super().__init__()
        
    def updateField(self, grid, gridOld, func, dt):
        """ 
        Can I overwrite this function?
        """

def RK4SchemeCoupled_OLD(funcs, grid, dt, nt):
    """ 
    Runge-Kutte-4 scheme for coupled equations. Old version.
    
    Inputs
    -------
    funcs : list of callable objects
            RHS of differenetial equation to be solved numerically.
    grid  : ArakawaCGrid object
            Object containing the domain and state information for the problem.
            Passed by reference.
    dt    : float
            Timestep.
    nt    : int 
            Current iteration of the simulation.
    
    Returns
    -------
    None
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
    grid.fields["eta"] += dt*(k1 + 2*k2 + 2*k3 + k4)/6
    grid.fields["uVelocity"] += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    grid.fields["vVelocity"] += dt*(m1 + 2*m2 + 2*m3 + m4)/6