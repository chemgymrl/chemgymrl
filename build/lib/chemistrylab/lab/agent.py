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
sys.path.append('../../chemistrylab/reactions')
import gym

# this is a class that is meant to interact with any of the lab benches or with the lab environment itself
class Agent:
    def __init__(self):
        self.name = 'agent'

    def run_step(self, env, spectra):
        """
        this is the function where the operation of your model or heuristic agent is defined
        """
        return np.array([])


class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = 'random'

    def run_step(self, env, spectra):
        return env.action_space.sample()

    def train_step(self, reward):
        pass
