"""

"""

import os
import sys

import numpy as np
import chemistrylab
import datetime as dt

Agent = object #TODO: add this actually

class Manager:
    def __init__(self, mode='custom', agent=None):
        """
        Constructor class module for the Manager class.

        Parameters
        ---------------
        'mode': `str`: ['human', 'random', 'custom']:
            Parameter to describe the mode that the manager will be run in. Note: `human` opens up a cli that can be
            used to run or by adding an agent to the agents.
        'agent': `agent.Agent`
            Parameter to specify a custom user made agent for the manager to use in solving the lab environment.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        self.mode = mode
        self.agents = {'random': RandomAgent}
        self.agent = agent
        self.lab = Lab()

    def register_agent(self, name: str, agent: Agent):
        """
        Allows the user to add a lab agent to the manager environment.
        Added lab agents can then use to perform actions in the lab environment.

        Parameters
        ---------------
        `name` : `str`
            The name of the agent to be registered.
        `agent` : `agent.Agent`
            The class representation of the agent to be registered.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # add the custom agent to the dictionary of registered agents
        self.agents[name] = agent

    def register_bench_agent(self, bench: str, name: str, agent: Agent):
        """
        Method to allow the user to add bench-specific agents to the lab environment.
        Added bench agents can then be used in order to perform actions in a bench

        Parameters
        ---------------
        `bench` : `str`
            The name of the bench to which the agent is being assigned.
        `name` : `str`
            The name of the agent to be registered.
        `agent` : `agent.Agent`
            The class representation of the agent to be registered.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # pass the parameters to the lab class to register the bench engine
        self.lab.register_agent(bench, name, agent)

    def run(self):
        """
        Method to run the specified agent or human mode based on the parameters passed to the Manager class.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # if "human" was specified, run the lab manager in human mode
        if self.mode == 'human':
            self._human_run()
        # if the mode given matches a registered agent, acquire the corresponding agent and run it
        elif self.mode in self.agents:
            self.agent = self.agents[self.mode]()
            self._agent_run()
        # if an agent was specified directly, run that agent
        elif self.agent:
            self.agent = self.agent()
            self._agent_run()
        else:
            raise ValueError("agent specified does not exist")

    def _human_run(self):
        """
        Method to run the specified agent or human mode based on the parameters passed to the Manager class.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # set up the done parameter
        done = False

        # specify the allowed human-specifiable commands
        commands = ['load vessel from pickle',
                    'load distillation bench',
                    'load reaction bench',
                    'load extraction bench',
                    'load characterization bench',
                    'list vessels',
                    'create new vessel',
                    'save vessel',
                    'quit']

        while not done:
            # print all the actions available
            print('Index: Action')
            for i, command in enumerate(commands):
                print(f'{i}: {command}')

            # instruct the user to select an action by means of its index
            action = int(input('Please select an action by index: '))

            # action[0] == load vessel from pickle file
            if action == 0:
                path = input('please specify the path of the vessel: ')
                self.load_vessel(path)
            # action[1] == load distillation bench
            elif action == 1:
                self._human_bench('distillation')
            # action[2] == load reaction bench
            elif action == 2:
                self._human_bench('reaction')
            # action[3] == load extraction bench
            elif action == 3:
                self._human_bench('extraction')
            # action[4] == load the analysis bench
            elif action == 4:
                self._human_bench('characterization')
                # action[5] == list the avilable vessels
            elif action == 5:
                self.list_vessels()
            # action[6] == create a new vessel
            elif action == 6:
                self.create_new_vessel()
            # action[7] == save an existing vessel
            elif action == 7:
                self.save_vessel()
            # action[8] OR action not in range(0, 8) == end the program
            else:
                done = True

    def _human_bench(self, bench: str):
        """
        Method to set up the bench, environment, vessel, and agent as specified by the human user.

        Parameters
        ---------------
        `bench` : `str`
            The name of the bench that has been selected in `_human_run`.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # acquire all the environments and agents from the selected bench
        if bench == 'distillation':
            envs = self.lab.distillations
            agents = list(self.lab.distill_agents.keys())
        elif bench == 'reaction':
            envs = self.lab.reactions
            agents = list(self.lab.react_agents.keys())
        elif bench == 'extraction':
            envs = self.lab.extractions
            agents = list(self.lab.extract_agents.keys())
        elif bench == 'characterization':
            envs = ['characterization']
            agents = ['none']
        else:
            raise KeyError('Inputted bench not supported')

        # print the environments of the selected bench and request the user select an environment
        print('index: env')
        for i, env in enumerate(envs):
            print(f'{i}: {env}')
        env = int(input('Please specify what environment you wish to use: '))

        # list all the vessels available on the shelf and request the user select a vessel
        self.list_vessels()
        vessel = int(input(
            'Please specify what vessel you wish to use (inputting -1 we will create a new vessel and use that) : '
        ))
        if vessel == -1:
            self.create_new_vessel()

        # list all the available agents and request the user select an agent
        for i, agent in enumerate(agents):
            print(f'{i}: {agent}')
        agent_ind = int(input('Please specify what agent you wish to use: '))

        # load the bench, environment, vessel, and agent and determine how much time it takes to do so
        start = dt.datetime.now()
        self.load_bench(bench, env, vessel, agent_ind)
        finish = dt.datetime.now()
        print(finish - start)

    def _agent_run(self):
        """
        Method to run the specified agent based on the parameters passed to the Manager class.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # reset the done parameter
        done = False

        # reset the lab environment
        self.lab.reset()

        # set up a reward counter
        total_reward = 0

        # set up an array to contain the analysis information made available to the lab manager
        analysis = np.array([])

        # have the agent select actions and run the lab manager step function until the done parameter is satisfied
        while not done:
            # agent selects actions based on the state of the environemnt and the chosen characterization of a vessel
            action = self.agent.run_step(self.lab, analysis)
            reward, analysis, done = self.lab.step(action)
            total_reward += reward

    def load_bench(self, bench, env_index, vessel_index, agent):
        """
        Method to load the specified bench based on the environment, vessel, and agent parameters.

        Parameters
        ---------------
        `bench` : `str`
            The name of the bench that is to be loaded.
        `env_index` : `np.int64`
            The index corresponding to the requested bench environment that is to be set up.
        `vessel_index` : `np.int64`
            The index corresponding to the requested vessel that is to be set up.
        `agent` : `np.int64`
            The index corresponding to the requested agent that is to be set up.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # pass the necessary parameters to the `run_bench` method of the `lab` environment
        self.lab.run_bench(bench, env_index, vessel_index, agent)

    def list_vessels(self):
        """
        Method to list all the vessels available on the shelf and their contents.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        for i, vessel in enumerate(self.lab.shelf.vessels):
            print(f'{i}: {vessel.label}')
            print(vessel.get_material_dict())
            print(vessel.get_solute_dict())

    def load_vessel(self, path):
        """
        Method to load a vessel from a specified path.

        Parameters
        ---------------
        `path` : `str`
            The directory path to a valid pickle file containing a vessel object.

        Returns
        ---------------
        None

        Raises
        ---------------
        `IOError`:
            Raised if the specified file path does not correspond to a valid vessel pickle file.
        """

        # check that the vessel path is a valid path
        if any([
                not os.path.isfile(path),
                path.split(".")[-1] != "pickle"
        ]):
            raise IOError("Invalid vessel path!")

        self.lab.shelf.load_vessel(path)

    def create_new_vessel(self):
        """
        Method to create a new vessel and place the new vessel on the shelf.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        self.lab.shelf.create_new_vessel()

    def save_vessel(self):
        """
        Method to save a human-specified vessel.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        """

        # set up a boolean to track if the specified parameters are completely validated
        valid_parameters = False

        # set up default vessel and path parameters
        vessel = 0
        path = ""

        # list the available vessels on the shelf and request a vessel to be saved in a certain location
        self.list_vessels()


        vessel = int(input('What vessel do you wish to save: '))
        path = input('Specify the relative path for the vessel: ')

        # pass the vessel and intended vessel path to the lab shelf
        self.lab.shelf.vessels[vessel].save_vessel(path)


if __name__ == "__main__":
    manager = Manager(mode='human')
    manager.run()
