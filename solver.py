"""
MTMW14 Assignment 2

Student ID: 31827379
"""

import numpy as np

# TODO: This was solver class for assignment 1, will need to be changed.

class Solver(object):
    
    def __init__(self, model, scheme, dt, nt, store=False):
        """ 
        """        
        self.model   = model
        self.scheme  = scheme
        self.dt      = dt
        self.nt      = nt
        
        self.store   = store
        self.history = None
        self.ensembleHistory = None
                
    def run(self, *phi0):
        """ 
        """
                              
        # Initialise storage arrays if necessary.
        if self.store:
            self.history = np.zeros(shape=(self.nt+1, len(self.model.eqns)))
            self.history[0, :] = phi0
                
        for i in range(1, self.nt+1):
                        
            # Calculate new time step values.
            phi = self.scheme(self.model.eqns, self.dt, i, *phi0)
            
            # Update previous time step values.
            phi0 = phi
            
            # Store results if necessary.
            if self.model.isScaled:
                phi = phi * np.array(self.model.dataScale)[:-1]
            
            if self.store:
                self.history[i, :] = phi
    
        return phi
    
    def runEnsemble(self, numEnsembles, perturbationRange, *phi0):
        """ 
        """
        
        # Copy - find better fix :(
        phi0 = phi0[:]
        
        # Overwrite store state if necessary (should be True in ensemble).
        store = self.store
        self.store = True
        
        # Initialise storage.
        self.ensembleHistory = []       # List for now.
        
        for _ in range(numEnsembles):
            
            # Non dimensionalise perturbation range.
            pert = np.array(perturbationRange) / np.array(self.scaleData)[:-1]
            
            # Perturb initial condition.
            phiPert = (np.array(phi0) + pert * (np.random.uniform(size=2) 
                                                * 2 - 1))
            
            # Run model.
            _ = self.run(*phiPert)
            
            # Store run.
            self.ensembleHistory.append(self.history)
        
        # Restore previous store state.
        self.store = store