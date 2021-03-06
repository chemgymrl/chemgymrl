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
import os


class TestVessel(TestCase):
    def test__update_temperature(self):
        vessel = Vessel("test")
        initial_temp = vessel.get_temperature()
        event = ['temperature change', 400, True]
        vessel.push_event_to_queue(feedback=[event], dt=0)
        self.assertEqual(vessel.get_temperature(), 400)

    def test__pour_by_volume(self):
        vessel = Vessel("test", materials={"H2O": [material.H2O, 100, 'mol']})
        initial_volume = vessel.get_current_volume()
        event = ['pour by volume', Vessel('test_2'), 0.1]
        vessel.push_event_to_queue(feedback=[event], dt=0)
        final_volume = vessel.get_current_volume()
        self.assertLess(final_volume[1], initial_volume[1])

    def test__drain_by_pixel(self):
        vessel = Vessel("test", materials={'C6H14': [material.C6H14, 1, 'mol'], 'H2O': [material.H2O, 1, 'mol']})
        initial_volume = vessel.get_current_volume()
        vessel2 = Vessel('test_2')
        event = ['drain by pixel', vessel2, 100]
        vessel.push_event_to_queue(events=[event], dt=10000)
        final_volume = vessel.get_current_volume()
        self.assertLess(final_volume[1], initial_volume[1])

    def test__drain_by_pixel_less(self):
        vessel = Vessel("test", materials={'C6H14': [material.C6H14, 1, 'mol'], 'H2O': [material.H2O, 1, 'mol']})
        initial_volume = vessel.get_current_volume()
        vessel2 = Vessel('test_2')
        event = ['drain by pixel', vessel2, 10]
        vessel.push_event_to_queue(events=[event], dt=1)
        final_volume = vessel.get_current_volume()
        self.assertLess(final_volume[1], initial_volume[1])

    def test__update_material_dict(self):
        vessel = Vessel("test")
        material_dict = {'H2O': [material.H2O, 100, 'mol'], 'C6H14': [material.C6H14, 30, 'mol']}
        event = ['update material dict', material_dict]
        vessel.push_event_to_queue(feedback=[event], dt=0)
        self.assertIn('H2O', vessel.get_material_dict())

    def test__update_solute_dict(self):
        vessel = Vessel("test", materials={'Na': [material.Na, 100, 'mol'], 'H2O': [material.H2O, 1, 'mol']})
        solute_dict = {'Na': {'H2O': [material.H2O, 100, 'mol']}}
        event = ['update solute dict', solute_dict]
        vessel.push_event_to_queue(feedback=[event], dt=0)
        self.assertIn('Na', vessel.get_solute_dict())

    def test__mix(self):
        vessel = Vessel("test", materials={'Na': [material.Na, 1, 'mol'], 'H2O': [material.H2O, 1, 'mol']})
        event = ['mix', 10]
        vessel.push_event_to_queue(feedback=[event], dt=0)

    def test_save_load(self):
        vessel = Vessel("test", materials={'Na': [material.Na, 1, 'mol'], 'H2O': [material.H2O, 1, 'mol']})
        vessel.save_vessel("test")
        new_vessel = Vessel("test")
        new_vessel.load_vessel("test.pickle")
        os.remove("test.pickle")

    def test_get_material_position(self):
        vessel = Vessel("test", materials={'C6H14': [material.C6H14, 1, 'mol'], 'H2O': [material.H2O, 1, 'mol']})
        vessel.get_material_position("H2O")

    def test_heat_change(self):
        vessel = Vessel("test", materials={'H2O': [material.H2O, 100, 'mol']})
        new_vessel = Vessel("test_2", materials={'H2O': [material.H2O, 100, 'mol']})
        temp = vessel.get_temperature()
        event = ["change_heat", 10, new_vessel]
        vessel.push_event_to_queue(feedback=[event], dt=0)
        self.assertLess(temp, vessel.get_temperature())

    def test_wait(self):
        vessel = Vessel("test", materials={'H2O': [material.H2O, 100, 'mol']})
        new_vessel = Vessel("test_2", materials={'H2O': [material.H2O, 100, 'mol']})
        temp = vessel.get_temperature()
        event = ["change_heat", 10, new_vessel]
        vessel.push_event_to_queue(feedback=[event], dt=0)
        self.assertLess(temp, vessel.get_temperature())
        event = ["wait", temp, new_vessel, True]
        vessel.push_event_to_queue(feedback=[event], dt=0)
        self.assertEqual(temp, vessel.get_temperature())
        event = ["wait", temp + 30, new_vessel, False]
        vessel.push_event_to_queue(feedback=[event], dt=10)
        self.assertLess(temp, vessel.get_temperature())

