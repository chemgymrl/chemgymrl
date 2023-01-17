"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import numpy as np
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.chem_algorithms import material, util
from chemistrylab.chem_algorithms.vessel import Vessel


class Shelf:
    """
    The shelf class is just as it sounds, it is a class to hold vessels from experiments
    """
    def __init__(self, max_num_vessels: int, vessel_paths: [np.array, list] = None):
        """
        initializes the shelf class
        Parameters
        ----------
        max_num_vessels: int: the maximum number of vessels that the shelf can store
        vessel_paths: a list of paths to vessel pickle files that are to be initialized into the shelf
        """
        # the next available slot for inserting a vessel
        self.open_slot = 0
        # maximum number of vessels that can be stored in the shelf
        self.max_num_vessels = max_num_vessels
        # a list that keeps all the vessels
        self.vessels = []
        if vessel_paths:
            for i, path in enumerate(vessel_paths):
                # function to load vessels from the path
                self.vessels[i] = Vessel(label=i)
                self.vessels[i].load_vessel(path)
            self.open_slot = len(vessel_paths)

    def create_new_vessel(self):
        """
        This function simple creates a new vessel and adds it to the shelf
        Returns
        -------
        None
        """
        index = self.open_slot
        assert index < self.max_num_vessels
        self.vessels.append(Vessel(label=index))
        self.open_slot += 1

    def _add_vessel_to_shelf(self, vessel):
        """
        This is a private member function that is used to add a vessel to the shelf
        Parameters
        ----------
        vessel: the vessel that is to be added to the shelf

        Returns
        -------
        None
        """
        assert self.open_slot < self.max_num_vessels
        vessel.label = self.open_slot
        self.vessels.append(vessel)
        self.open_slot += 1

    def get_vessel(self, index):
        """
        a function that returns a vessel from the shelf and removes it from the shelf
        Parameters
        ----------
        index: the index of the vessel that the user wishes to retrieve from the shelf

        Returns
        -------
        the vessel at the specified index or the vessel that was last added to the shelf, if there are
        no vessels in the shelf then it creates a vessel and returns it
        """
        if self.open_slot == 0:
            self.create_new_vessel()
            self.open_slot -= 1
            return self.vessels.pop(0)
        elif index > self.open_slot or index >= len(self.vessels):
            self.open_slot -= 1
            return self.vessels.pop(-1)
        elif index < len(self.vessels):
            self.open_slot -= 1
            return self.vessels.pop(index)
        else:
            raise ValueError('you failed to consider a case nicholas')


    def delete_vessel(self, index):
        """
        deletes a vessel from the shelf at the specified index

        Parameters
        ----------
        index: the index of the vessel which is to be deleted

        Returns
        -------
        None
        """
        if index < self.open_slot:
            self.open_slot -= 1
            self.vessels.pop(index)
        else:
            assert KeyError(f"{index} out of arrange for array of size {len(self.vessels)}")

    def return_vessel_to_shelf(self, vessel: [Vessel, list]):
        """
        This function allows the user to add a vessel to the shelf after an experiment has been conducted

        Parameters
        ----------
        vessel: [Vessel, list]: a vessel or list of vessels that are to be added to the shelf

        Returns
        -------
        None

        """
        if type(vessel) == list:
            for ves in vessel:
                if type(ves) != Vessel:
                    raise TypeError(f'{type(vessel)} is not supported')
                self._add_vessel_to_shelf(ves)
        elif type(vessel) == Vessel:
            self._add_vessel_to_shelf(vessel)
        else:
            raise TypeError(f'{type(vessel)} is not supported')

    def load_vessel(self, path: str):
        """
        a function that loads a vessel from a pickle file and places it into the shelf
        Parameters
        ----------
        path: the path to the vessel that is to be loaded

        Returns
        -------
        None
        """
        self.create_new_vessel()
        self.vessels[-1].load_vessel(path)

    def reset(self):
        """
        a function that deletes all vessels in the shelf and resets the next slot to insert

        Returns
        -------
        None
        """
        while len(self.vessels) > 0:
            self.delete_vessel(0)
