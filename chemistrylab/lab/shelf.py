import sys
import numpy as np
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.chem_algorithms import material, util
from chemistrylab.chem_algorithms.vessel import Vessel


class Shelf:
    def __init__(self, max_num_vessels, vessel_paths=None):
        self.open_slot = 0
        self.max_num_vessels = max_num_vessels
        self.vessels = []
        if vessel_paths:
            for i, path in enumerate(vessel_paths):
                # function to load vessels from the path
                self.vessels[i] = Vessel(label=i)

    def create_new_vessel(self):
        index = self.open_slot
        assert index < self.max_num_vessels
        self.vessels.append(Vessel(label=index))
        self.open_slot += 1

    def _add_vessel_to_shelf(self, vessel):
        vessel.label = self.open_slot
        self.vessels.append(vessel)
        self.open_slot += 1

    def get_vessel(self, index):
        self.open_slot -= 1
        return self.vessels.pop(index)

    def delete_vessel(self, index):
        self.open_slot -= 1
        self.vessels.pop(index)

    def return_vessel_to_shelf(self, vessel):
        if type(vessel) == list:
            for ves in vessel:
                if type(ves) != Vessel:
                    raise TypeError(f'{type(vessel)} is not supported')
                self._add_vessel_to_shelf(ves)
        elif type(vessel) == Vessel:
            self._add_vessel_to_shelf(vessel)
        else:
            raise TypeError(f'{type(vessel)} is not supported')

    def load_vessel(self, path):
        self.create_new_vessel()
        self.vessels[-1].load_vessel(path)

    def reset(self):
        while len(self.vessels) > 0:
            self.delete_vessel(0)
