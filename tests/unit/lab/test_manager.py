from unittest import TestCase
import sys
import gym
sys.path.append('../../')
from chemistrylab.lab.manager import Manager
from chemistrylab.lab.agent import RandomAgent
import numpy as np


class TestManager(TestCase):
    def test_register_agent(self):
        manager = Manager()
        manager.register_agent("test", RandomAgent)
        self.assertIn("test", manager.agents)

    def test_run(self):
        manager = Manager(mode='random')
        manager.run()
