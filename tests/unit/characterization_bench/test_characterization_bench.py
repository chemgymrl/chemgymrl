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
sys.path.append("../../") # to access `chemistrylab`
from unittest import TestCase
from chemistrylab.characterization_bench.characterization_bench import CharacterizationBench
from chemistrylab.chem_algorithms import material, util
from chemistrylab.chem_algorithms.vessel import Vessel
import numpy as np


class TestCharacterizationBench(TestCase):
    def test_analyze(self):
        char_bench = CharacterizationBench()
        vessel = Vessel("test", materials={'Na': [material.Na, 1, 'mol']})
        analysis = char_bench.analyze(vessel, 'spectra')
        self.assertEqual(len(analysis[0]), 200)

