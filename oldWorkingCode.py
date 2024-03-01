"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np
import matplotlib.pyplot as plt

# DEFAULT PARAMETERS
f0 = 1e-4    # Coriolis parameter [s^-1]
beta = 1e-11  # Coriolis parameter gradient [m^-1s^-1] 
g = 10        # Gravitational acceleration [ms^-2]
gamma = 1e-6 # Linear drag coefficient [s^-1]
rho = 1000    # Density of water [kgm^-3]
H = 1000      # Height [m]

tau0 = 0.2   # Wind stress amplitude [Nm^-2]

# GRID SPECIFICATION
L = 1e6  # Grid length [m]
d = 50e3  # Grid spacing (equal in x and y dimensions) [m]
nx = int(L/d)   # No. of grid spaces in x and y directions.
ny = nx    # Same no. of spaces in x and y directions.

#%% CALCULATE THE ANALYTICAL SOLUTION 

eta0 = 0

# Input arguments
x = 0
y = 0

# Calculate terms used in solution.
epsilon = gamma / (L * beta)
a = (-1 - np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)
b = (-1 + np.sqrt(1 + (2*np.pi*epsilon)**2)) / (2*epsilon)

f1 = lambda x: np.pi*(1 + ((np.exp(a) - 1)*np.exp(b*x) + 
                     (1 - np.exp(b))*np.exp(a*x))/(np.exp(b) - np.exp(a)))

f2 = lambda x: ((np.exp(a) - 1)*b*np.exp(b*x) + 
                     (1 - np.exp(b))*a*np.exp(a*x))/(np.exp(b) - np.exp(a))

# Calculate the velocities and height perturbation.
u = lambda x, y: - tau0 / (np.pi*gamma*rho*H) * f1(x/L)*np.cos(np.pi*y/L)
v = lambda x, y: tau0/(np.pi*gamma*rho*H) * f2(x/L)*np.sin(np.pi*y/L)
eta = lambda x, y: eta0 + tau0/(np.pi*gamma*rho*H)*f0*L/g * (gamma/(f0*np.pi)*f2(x/L)*np.cos(np.pi*y/L) 
                                                             + f1(x/L)/np.pi*(np.sin(np.pi*y/L)*(1 + beta*y/f0) + beta*L/(f0*np.pi)*np.cos(np.pi*y/L)))

#%%

# Mesh grid
x0 = 0
pointsX = np.linspace(x0, L, nx+1)
pointsY = pointsX.copy()                 # Just in case for future.
XS, YS = np.meshgrid(pointsX, pointsY)

uSol = u(XS, YS)
vSol = v(XS, YS)
etaSol = eta(XS, YS)

fig, axs = plt.subplots(1, 3, figsize=(32, 13))

cont1 = axs[0].contourf(XS, YS, uSol, levels=25)
plt.colorbar(cont1, location='bottom')
axs[0].set_xlabel("X", fontsize=25)
axs[0].set_ylabel("Y", fontsize=25)
axs[0].set_title("u", fontsize=25)
    
cont2 = axs[1].contourf(XS, YS, vSol, levels=25)
plt.colorbar(cont2, location='bottom')
axs[1].set_xlabel("X", fontsize=25)
axs[1].set_title("v", fontsize=25)

cont3 = axs[2].contourf(XS, YS, etaSol, levels=25)
plt.colorbar(cont3, location='bottom')
# axs[2].contour(XS, YS, uSol, colors='black')
axs[2].set_xlabel("X", fontsize=25)
axs[2].set_title("$\eta$", fontsize=25)

plt.show()

#%% TASK D

# Energy function for TASK E
def calculateEnergy(u, v, eta):
    return np.sum(0.5*rho*(u[:, :-1]**2 + v[:-1, :]**2 + g*eta**2))*d**2

# TODO: obvs clean this up

# DOn't need two functions here.
def interpolateV(v):
    return (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])/4

def interpolateU(u):
    return (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])/4

# Calculate wind stress.
tauX = tau0 * - np.cos(np.pi*YS/L)
tauY = np.zeros_like(tauX)

# Set up c grid fields (already includes boundary and initial conditions)
uField = np.zeros(shape=(ny, nx+1))
vField = np.zeros(shape=(ny+1, nx))
hField = np.zeros(shape=(nx, ny))

# # Calculate the Gaussian perturbation
# h_perturbation = 5 * np.exp(-((XS[:-1, :-1] - 5e5)**2 / (2 * 50e3**2) + (YS[:-1, :-1] - 5e5)**2 / (2 * 50e3**2)))

# # Add the perturbation to the initial condition
# hField += h_perturbation

#%%

dt = 350  # seconds
nt = int(np.ceil(10*24*60**2/dt))

# Coriolis parameter (f=f0 at y=0).
f = f0 + beta*YS

hState1 = []
uState1 = []
for t in range(nt):
    
    # Calculate velocity gradients throughout the domain.
    dudx = (uField[:, 1:] - uField[:, :-1]) / d
    dvdy = (vField[1:, :] - vField[:-1, :]) / d
        
    # Calculate new eta.
    hField2 = hField - H*dt*(dudx + dvdy)
    
    # Calculate eta derivates for updated time step.
    detadx = (hField2[:, 1:] - hField2[:, :-1])/d
    detady = (hField2[1:, :] - hField2[:-1, :])/d
    
    if t % 2 == 0:
    # if t != -1:
        # Interploate v on u-grid.
        vInterp = interpolateV(vField)
        
        # Calculate new u.
        uField2 = uField.copy()  # Include the boundary conditions.
        uField2[:, 1:-1] += dt*(f[:-1, 1:-1]*vInterp - g*detadx - 
                                gamma*uField[:, 1:-1] + 
                                tauX[:-1, 1:-1]/(rho*H))
        
        
        # Update interpolated values of u.
        uInterp = interpolateU(uField2)
        
        # Calculate new v.
        vField2 = vField.copy() # Include the boundary conditions.
        vField2[1:-1, :] += dt*(-f[1:-1,:-1]*uInterp - g*detady - 
                                gamma*vField[1:-1, :] 
                                # + 
                                # tauY[1:-1, np.newaxis]/(rho*H)
                                )
    else:
        
        # Interpolate u on v-grid.
        uInterp = interpolateU(uField)
        
        # Calculate new v.
        vField2 = vField.copy() # Include the boundary conditions.
        vField2[1:-1, :] += dt*(-f[1:-1,:-1]*uInterp - g*detady - 
                                gamma*vField[1:-1, :] 
                                # + 
                                # tauY[1:-1, np.newaxis]/(rho*H)
                                )
        
        # Update interpolated values of v.
        vInterp = interpolateV(vField2)
        
        # Calculate new u.
        uField2 = uField.copy()  # Include the boundary conditions.
        uField2[:, 1:-1] += dt*(f[:-1, 1:-1]*vInterp - g*detadx - 
                                gamma*uField[:, 1:-1] + 
                                tauX[:-1, 1:-1]/(rho*H))
        
    # Update values.
    hField = hField2 
    uField = uField2 
    vField = vField2
    
    uState1.append(uField)
    hState1.append(hField)
        
    # Plot contours
    fig, axs = plt.subplots(1, 3, figsize=(32, 13))
    
    cont1 = axs[0].imshow(uField)
    plt.colorbar(cont1, location='bottom')
    axs[0].set_xlabel("X", fontsize=25)
    axs[0].set_ylabel("Y", fontsize=25)
    axs[0].set_title("u", fontsize=25)
    
    cont2 = axs[1].imshow(vField)
    plt.colorbar(cont2, location='bottom')
    axs[1].set_xlabel("X", fontsize=25)
    axs[1].set_title("v", fontsize=25)
    
    cont3 = axs[2].imshow(hField)
    plt.colorbar(cont3, location='bottom')
    # axs[2].contour(XS, YS, uSol, colors='black')
    axs[2].set_xlabel("X", fontsize=25)
    axs[2].set_title("$\eta$", fontsize=25)
    
    plt.show()

#%% Plot the thing
fig, axs = plt.subplots(2, 2, figsize=(24, 24))

# u vs x along the grid, closest to the southern edge of the basin.
axs[0, 0].plot(XS[0, :], uField[0, :])
axs[0, 0].set_title('u vs x')

# v vs y along the grid, closest to the western edge of the basin.
axs[0, 1].plot(YS[:, 0], vField[:, 0])
axs[0, 1].set_title('v vs y')

# eta vs x through the middle of the gyre.
axs[1, 0].plot(XS[0, :-1], hField[10, :])
axs[1, 0].set_title('eta vs x')

# 2D contour plot showing elevation eta.
axs[1, 1].imshow(hField, extent=[XS.min(), XS.max(), YS.min(), YS.max()], origin='lower', aspect='auto')
axs[1, 1].set_title('2D Contour Plot')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('y')

# Add colorbar for the 2D contour plot
cbar = plt.colorbar(axs[1, 1].imshow(hField, origin='lower', aspect='auto'), ax=axs[1, 1])
cbar.set_label('Elevation (eta)')

plt.tight_layout()
plt.show()