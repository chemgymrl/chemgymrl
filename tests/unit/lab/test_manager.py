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
        for i in range(100):
            print("----------------")
            print(i)
            print("----------------")
            manager = Manager(mode='random')
            manager.run()
