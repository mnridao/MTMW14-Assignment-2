"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import matplotlib.pyplot as plt
    
def plotContourSubplot(u, v, eta):
    """ 
    Contour plots of each of the fields stored in the grid.
    
    Inputs
    -------
    u   : np array
          Array containing the u-velocity field.
    v   : np array
          Array containing the v-velocity field.
    eta : np array
         Array containing the height perturbation field (eta).
    """
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 8))
    
    # Subplot for u-velocity solution.
    cont1 = axs[0].imshow(u)
    plt.colorbar(cont1, location='bottom')
    axs[0].set_xlabel("X", fontsize=20)
    axs[0].set_ylabel("Y", fontsize=20)
    axs[0].set_title("u", fontsize=20)
    axs[0].tick_params(labelsize=15)
        
    # SUbplot for v-velocity solution.
    cont2 = axs[1].imshow(v)
    plt.colorbar(cont2, location='bottom')
    axs[1].set_xlabel("X", fontsize=20)
    axs[1].set_title("v", fontsize=20)
    axs[1].tick_params(labelsize=15)

    # Subplot for eta solution.
    cont3 = axs[2].imshow(eta)
    plt.colorbar(cont3, location='bottom')
    axs[2].set_xlabel("X", fontsize=20)
    axs[2].set_title("$\eta$", fontsize=20)
    axs[2].tick_params(labelsize=15)

    plt.show()