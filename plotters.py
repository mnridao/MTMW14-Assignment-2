"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import matplotlib.pyplot as plt

def plotContourSubplot(grid):
    """ 
    """
    fig, axs = plt.subplots(1, 3, figsize=(32, 13))
    
    # Subplot for u-velocity solution.
    cont1 = axs[0].imshow(grid.uField)
    plt.colorbar(cont1, location='bottom')
    axs[0].set_xlabel("X", fontsize=25)
    axs[0].set_ylabel("Y", fontsize=25)
    axs[0].set_title("u", fontsize=25)
        
    # SUbplot for v-velocity solution.
    cont2 = axs[1].imshow(grid.vField)
    plt.colorbar(cont2, location='bottom')
    axs[1].set_xlabel("X", fontsize=25)
    axs[1].set_title("v", fontsize=25)

    # Subplot for eta solution.
    cont3 = axs[2].imshow(grid.hField)
    plt.colorbar(cont3, location='bottom')
    axs[2].set_xlabel("X", fontsize=25)
    axs[2].set_title("$\eta$", fontsize=25)

    plt.show()
    
# Quiver plot + height perturbation plot (with fixed colorbar and time shown).

