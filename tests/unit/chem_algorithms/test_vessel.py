from unittest import TestCase
import sys
import numpy as np
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.chem_algorithms import material, util
from chemistrylab.chem_algorithms.vessel import Vessel


class TestVessel(TestCase):
    def test__update_temperature(self):
        vessel = Vessel("test")
        event = []
        vessel.push_event_to_queue()

    def test__change_heat(self):
        self.fail()

    def test__pour_by_volume(self):
        self.fail()

    def test__drain_by_pixel(self):
        self.fail()

    def test__fully_mix(self):
        self.fail()

    def test__update_material_dict(self):
        self.fail()

    def test__update_solute_dict(self):
        self.fail()

    def test__mix(self):
        self.fail()

    def test__update_layers(self):
        self.fail()
