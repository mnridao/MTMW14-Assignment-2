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