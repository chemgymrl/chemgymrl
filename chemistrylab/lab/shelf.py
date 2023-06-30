"""

"""
import sys
import numpy as np
from chemistrylab import material
from chemistrylab.vessel import Vessel
from copy import deepcopy

class Shelf:
    """
    The shelf class holds vessels from experiments.
    
    pop, __getitem__, __delitem__, and append are implemented so shelves can be used similar to a list of vessels
    """
    def __init__(self, vessels,n_working = 1):
        """
        TODO: Allow the starting vessels to be given as arguments.
        """
        self._orig_vessels=[v for v in vessels]
        assert n_working<=len(self._orig_vessels)
        self.n_working = n_working
        self.reset()

    def get_working_vessels(self):
        """Returns a tuple of the 'working' vessels used for observations and rewards."""
        return tuple(self.vessels[:self.n_working])
    def get_vessels(self):
        return tuple(self.vessels)
    def pop(index=-1):
        return self.vessels.pop(index)
    def __getitem__(self, slice):
        return self.vessels[slice]
    def __setitem__(self, slice, item):
        self.vessels[slice] = item
    def __delitem__(self, slice):
        del self.vessels[slice]
    def __len__(self):
        return len(self.vessels)
    def append(self,vessel: Vessel):
        self.vessels.append(vessel)
    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + ': (%s)' % ", ".join(str(v) for v in self.vessels)

    def load_vessel(self, path: str):
        """
        TODO: Implement this
        Args:
        - path (str): the path to the vessel that is to be loaded
        """
        pass

    def reset(self, target = None):
        """
        TODO: Update this along with __init__
        Resets the shelf to it's initial state
        """
        self.vessels=[deepcopy(v) for v in self._orig_vessels]



class VariableShelf(Shelf):
    """
    Shelf which is given a set of fixed and variable vessels. 
    On reset the original vessels are deepcopied into new vessel objects
    
    """
    def __init__(self, variable_vessels: list, fixed_vessels: list, n_working = 1):
        """
        TODO: Allow the starting vessels to be given as arguments.
        """
        self.n_working = n_working
        self.variable_vessels = variable_vessels
        self.fixed_vessels = fixed_vessels

        self._length = len(variable_vessels)+len(fixed_vessels)
        assert n_working<=self._length

        self.reset()

    def reset(self, target = None):

        variable = []
        if len(self.variable_vessels)>0:
            # Set the target to the first one in the dict if not provided
            variable = [vessel_func(target) for vessel_func in self.variable_vessels]
        
        self.vessels = variable + [deepcopy(v) for v in self.fixed_vessels]

