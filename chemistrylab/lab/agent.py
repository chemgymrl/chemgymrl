import sys
import numpy as np
sys.path.append("../../") # to access `chemistrylab`
sys.path.append('../../chemistrylab/reactions')
import gym


class Agent:
    def __init__(self):
        self.name = 'agent'

    def run(self, env, state):
        return np.array([])


class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = 'random'

    def run_step(self, env, state):
        return env.action_space.sample()
