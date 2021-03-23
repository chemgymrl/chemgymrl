import sys
import numpy as np
sys.path.append("../../") # to access `chemistrylab`
sys.path.append('../../chemistrylab/reactions')
import gym

# this is a class that is meant to interact with any of the lab benches or with the lab environment itself
class Agent:
    def __init__(self):
        self.name = 'agent'

    def run_step(self, env, state):
        """
        this is the function where the operation of your model or heuristic agent is defined
        """
        return np.array([])


class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = 'random'

    def run_step(self, env, state):
        return env.action_space.sample()
