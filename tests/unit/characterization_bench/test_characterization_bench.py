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
        self.assertEqual(len(analysis), 200)

