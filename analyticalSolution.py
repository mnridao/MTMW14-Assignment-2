"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

""" 
"""
f1 = lambda x, a, b: np.pi*(1 + ((np.exp(a) - 1)*np.exp(b*x) + 
                     (1 - np.exp(b))*np.exp(a*x))/(np.exp(b) - np.exp(a)))

f2 = lambda x, a, b: ((np.exp(a) - 1)*b*np.exp(b*x) + 
                     (1 - np.exp(b))*a*np.exp(a*x))/(np.exp(b) - np.exp(a))

def analyticalSolution(X, Y, L, params):
    """ 
    """ 
    
    epsilon = params.gamma / (L * params.beta)
    a = (-1 - np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    b = (-1 + np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
    
    # Calculate u-velocity.
    u = (- params.tau0 / (np.pi*params.gamma*params.rho*params.H) * 
         f1(X/L, a, b) * np.cos(np.pi*Y/L))
    
    # Calculate v-velocity.
    v = (params.tau0/(np.pi*params.gamma*params.rho*params.H) * 
         f2(X/L, a, b) * np.sin(np.pi*Y/L))
    
    # Calculate eta.
    eta = (params.tau0/(np.pi*params.gamma*params.rho*params.H) * 
           params.f0*L/params.g * (params.gamma/(params.f0*np.pi) * 
           f2(X/L, a, b)*np.cos(np.pi*Y/L) + f1(X/L, a, b)/np.pi * 
           (np.sin(np.pi*Y/L)*(1 + params.beta*Y/params.f0) + params.beta*L /
           (params.f0*np.pi)*np.cos(np.pi*Y/L))))
    
    return u, v, eta   # Don't really like this.

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    from grids import ArakawaCGrid 
    from equations import Parameters
    
    # Set up the grid.
    x0 = 0
    xL = 1e6
    d = 50e3
    nx = int(xL/d)
    grid = ArakawaCGrid([x0, xL], nx)
    
    # Get the default parameters.
    params = Parameters()
    
    # Calculate and plot analytical solutions.
    u, v, eta = analyticalSolution(grid.X, grid.Y, xL, params)
    
    fig, axs = plt.subplots(1, 3, figsize=(32, 13))

    cont1 = axs[0].contourf(grid.X, grid.Y, u, levels=25)
    plt.colorbar(cont1, location='bottom')
    axs[0].set_xlabel("X", fontsize=25)
    axs[0].set_ylabel("Y", fontsize=25)
    axs[0].set_title("u", fontsize=25)
        
    cont2 = axs[1].contourf(grid.X, grid.Y, v, levels=25)
    plt.colorbar(cont2, location='bottom')
    axs[1].set_xlabel("X", fontsize=25)
    axs[1].set_title("v", fontsize=25)

    cont3 = axs[2].contourf(grid.X, grid.Y, eta, levels=25)
    plt.colorbar(cont3, location='bottom')
    # axs[2].contour(XS, YS, uSol, colors='black')
    axs[2].set_xlabel("X", fontsize=25)
    axs[2].set_title("$\eta$", fontsize=25)

    plt.show()