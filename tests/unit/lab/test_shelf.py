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
from unittest import TestCase
import sys
import numpy as np
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.chem_algorithms import material, util
from chemistrylab.chem_algorithms.vessel import Vessel
from chemistrylab.lab.shelf import Shelf


class TestShelf(TestCase):
    def test_create_new_vessel(self):
        shelf = Shelf(100)
        shelf.create_new_vessel()
        self.assertEqual(shelf.open_slot, 1)
        self.assertEqual(len(shelf.vessels), 1)

    def test_get_vessel(self):
        shelf = Shelf(100)
        shelf.create_new_vessel()
        shelf.get_vessel(0)

        shelf = Shelf(100)
        shelf.get_vessel(0)

    def test_delete_vessel(self):
        shelf = Shelf(100)
        shelf.create_new_vessel()
        shelf.delete_vessel(0)
        self.assertEqual(shelf.open_slot, 0)
        self.assertEqual(len(shelf.vessels), 0)

    def test_return_vessel_to_shelf(self):
        test_vessel = Vessel('test')
        shelf = Shelf(100)
        shelf.return_vessel_to_shelf(test_vessel)
        self.assertEqual(shelf.open_slot, 1)
        self.assertEqual(len(shelf.vessels), 1)

    def test_reset(self):
        shelf = Shelf(100)
        for i in range(10):
            shelf.create_new_vessel()
        shelf.reset()
        self.assertEqual(shelf.open_slot, 0)
        self.assertEqual(len(shelf.vessels), 0)
