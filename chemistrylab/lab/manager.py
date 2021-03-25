import sys
from abc import ABC

import numpy as np
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.lab.lab import Lab
from chemistrylab.lab.agent import RandomAgent
import datetime as dt
import gym

"""
The manager class works as an interface for the agent to work with the lab environment and helps to automate the use of 
the entire lab
"""


class Manager:
    def __init__(self, mode='custom', agent=None):
        """
        'mode': str: ['human', 'random', 'custom']: describes the mode that the manager will be run in, human opens up a
         cli that can be used to run or by adding an agent to the agents
         'agent': Agent: specifies a custom user made agent for the manager to use in solving the lab environment
        """

        self.mode = mode
        self.agents = {'random': RandomAgent}
        self.agent = agent
        self.lab = Lab()

    def register_agent(self, name, agent):
        """
        allows the user to add an agent to the manager environment which they can then use to perform actions in the lab
        environment
        """
        self.agents[name] = agent

    def register_bench_agent(self, bench, name, agent):
        """
        allows the user to add agents to the lab environment which can then be used in order to perform actions in a
        bench
        """
        self.lab.register_agent(bench, name, agent)

    def run(self):
        if self.mode == 'human':
            self._human_run()
        elif self.mode in self.agents:
            self.agent = self.agents[self.mode]()
            self._agent_run()
        elif self.agent:
            self._agent_run()
        else:
            raise ValueError("agent specified does not exist")

    def _human_run(self):
        done = False
        commands = ['load vessel from pickle',
                    'load distillation bench',
                    'load reaction bench',
                    'load extraction bench',
                    'load analysis bench',
                    'list vessels',
                    'create new vessel',
                    'save vessel',
                    'quit']
        while not done:
            print('Index: Action')
            for i, command in enumerate(commands):
                print(f'{i}: {command}')
            action = int(input('Please select an action: '))

            if action == 0:
                path = input('please specify the path of the vessel: ')
                self.load_vessel(path)
            elif action == 1:
                self._human_bench('distillation')
            elif action == 2:
                self._human_bench('reaction')
            elif action == 3:
                self._human_bench('extraction')
            elif action == 4:
                self._human_bench('analysis')
            elif action == 5:
                self.list_vessels()
            elif action == 6:
                self.create_new_vessel()
            elif action == 7:
                self.save_vessel()
            else:
                done = True

    def _human_bench(self, bench):
        if bench == 'distillation':
            envs = self.lab.distillations
            agents = list(self.lab.distill_agents.keys())
        elif bench == 'reaction':
            envs = self.lab.reactions
            agents = list(self.lab.react_agents.keys())
        elif bench == 'extraction':
            envs = self.lab.extractions
            agents = list(self.lab.extract_agents.keys())
        elif bench == 'analysis':
            envs = ['analysis']
            agents = ['none']
        else:
            raise KeyError('bench not supported')
        print('index: env')
        for i, env in enumerate(envs):
            print(f'{i}: {env}')
        env = int(input('Please specify what environment you wish to use: '))
        self.list_vessels()
        vessel = int(input(
            'Please specify what vessel you wish to use (if you input -1 we will create a new vessel and use that) : '))
        if vessel == -1:
            self.create_new_vessel()
        for i, agent in enumerate(agents):
            print(f'{i}: {agent}')
        agent_ind = int(input('Please specify what agent you wish to use: '))
        start = dt.datetime.now()
        self.load_bench(bench, env, vessel, agent_ind)
        finish = dt.datetime.now()
        print(finish - start)

    def _agent_run(self):
        done = False
        self.lab.reset()
        total_reward = 0
        analysis = np.array([])
        while not done:
            action = self.agent.run_step(self.lab, analysis)
            print(action)
            reward, analysis, done = self.lab.step(action)
            total_reward += reward

    def load_bench(self, bench, env_index, vessel_index, agent):
        self.lab.run_bench(bench,  env_index, vessel_index, agent)

    def list_vessels(self):
        for i, vessel in enumerate(self.lab.shelf.vessels):
            print(f'{i}: {vessel.label}')
            print(vessel.get_material_dict())
            print(vessel.get_solute_dict())

    def load_vessel(self, path):
        self.lab.shelf.load_vessel(path)

    def create_new_vessel(self):
        self.lab.shelf.create_new_vessel()

    def save_vessel(self):
        self.list_vessels()
        vessel = int(input('What vessel do you wish to save: '))
        path = input('please specify the relative path for the vessel: ')
        self.lab.shelf.vessels[vessel].save_vessel(path)


if __name__ == "__main__":
    manager = Manager(mode='human')
    manager.run()