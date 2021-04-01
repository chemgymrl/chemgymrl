from unittest import TestCase
import sys
import gym
sys.path.append('../../')
from chemistrylab.lab.lab import Lab
from chemistrylab.lab.agent import RandomAgent
import numpy as np

class TestLab(TestCase):
    def test_register_agent(self):
        lab = Lab()
        lab.register_agent("reaction", "test", RandomAgent)
        self.assertIn("test", lab.react_agents)

    def test_load_reaction_bench(self):
        lab = Lab()
        env = lab.load_reaction_bench(1)
        self.assertEqual(env.__class__.__bases__[0].__name__, 'Wrapper')

    def test_load_extraction_bench(self):
        lab = Lab()
        env = lab.load_extraction_bench(0)
        self.assertEqual(env.__class__.__bases__[0].__name__, 'Wrapper')

    def test_load_distillation_bench(self):
        lab = Lab()
        env = lab.load_distillation_bench(0)
        self.assertEqual(env.__class__.__bases__[0].__name__, 'Wrapper')

    def test_step(self):
        lab = Lab()
        react_action = np.array([0, 1, 0, 0])
        extract_action = np.array([1, 0, 0, 0])
        distill_action = np.array([2, 0, 2, 0])
        analysis_action = np.array([3, 0, 2, 0])
        lab.step(react_action)
        lab.step(extract_action)
        lab.step(distill_action)
        lab.step(analysis_action)


    def test_reset(self):
        lab = Lab()
        react_action = np.array([0, 1, 0, 0])
        lab.step(react_action)
        lab.reset()
        self.assertEqual(len(lab.shelf.vessels), 0)
