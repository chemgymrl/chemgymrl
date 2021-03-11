import sys
import numpy as np
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.lab.lab import Lab
import datetime as dt


class Manager:
    def __init__(self, mode='human'):
        self.mode = mode
        self.agent = None
        self.lab = Lab()

    def run(self):
        if self.mode == 'human':
            self._human_run()

    def _human_run(self):
        done = False
        commands = ['load vessel from pickle',
                    'load distillation bench',
                    'load reaction bench',
                    'load extraction bench',
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
                self.list_vessels()
            elif action == 5:
                self.create_new_vessel()
            elif action == 6:
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
        agent = agents[agent_ind]
        start = dt.datetime.now()
        self.load_bench(bench, env, vessel, agent)
        finish = dt.datetime.now()
        print(finish - start)

    def _agent_run(self):
        self.load_agent()

    def load_agent(self):
        pass

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
    manager = Manager()
    manager.run()
