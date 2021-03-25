import sys
from abc import ABC

sys.path.append("../../") # to access `chemistrylab`
sys.path.append('../../chemistrylab/reactions')
from chemistrylab.lab.agent import RandomAgent
from chemistrylab.lab.shelf import Shelf
import gym
import chemistrylab
import numpy as np
from gym import envs
from chemistrylab.analysis_bench.analysis_bench import AnalysisBench


class Lab(gym.Env, ABC):
    """
    The lab class is meant to be a gym environment so that an agent can figure out how to synthesize different chemicals
    """
    def __init__(self, render_mode=None, max_num_vessels=100):
        all_envs = envs.registry.all()
        # the following parameters list out all available reactions, extractions and distillations that the agent can use
        self.reactions = [env_spec.id for env_spec in all_envs if 'React' in env_spec.id]
        self.extractions = [env_spec.id for env_spec in all_envs if 'Extract' in env_spec.id]
        self.distillations = [env_spec.id for env_spec in all_envs if 'Distill' in env_spec.id]
        self.analysis = list(AnalysisBench().techniques.keys())
        self.analysis_bench = AnalysisBench()
        # the following is a dictionary of all available agents that can operate each bench feel free to add your own
        # custom agents
        self.react_agents = {'random': RandomAgent()}
        self.extract_agents = {'random': RandomAgent()}
        self.distill_agents = {'random': RandomAgent()}
        self.render_mode = render_mode
        self.max_num_vessels = max_num_vessels
        # the shelf holds all available vessels that can be used by the agent
        self.shelf = Shelf(max_num_vessels=max_num_vessels)
        # the action space is a vector:
        # the 0th index represents what bench is selected (reaction, extraxction, distillation, done)
        # the 1st index represents what environment the agent selects
        # the 2nd index represents what vessel from the shelf the agent uses
        # the 3rd index represents which agent will be used to perform the experiment
        self.action_space = gym.spaces.MultiDiscrete([4,
                                                      max([len(self.reactions),
                                                           len(self.extractions),
                                                           len(self.distillations),
                                                           len(self.analysis)]),
                                                      self.max_num_vessels,
                                                      max([len(self.react_agents),
                                                           len(self.extract_agents),
                                                           len(self.distill_agents)])])
        # have to get back to this part
        self.observation_space = None

    def register_agent(self, bench, name, agent):
        """
        a function that allows the user to add their own custom agents to a bench
        """
        if bench == "reaction":
            self.react_agents[name] = agent
        elif bench == "extraction":
            self.extract_agents[name] = agent
        elif bench == "distillation":
            self.distill_agents[name] = agent
        else:
            raise ValueError('bench must be "reaction", "extraction" or "distillation"')

        self.action_space = gym.spaces.MultiDiscrete([4,
                                                      max([len(self.reactions),
                                                           len(self.extractions),
                                                           len(self.distillations)]),
                                                      self.max_num_vessels,
                                                      max([len(self.react_agents),
                                                           len(self.extract_agents),
                                                           len(self.distill_agents)])])

    def load_reaction_bench(self, index):
        # this function returns the reaction environment that the agent has selected
        print(self.reactions[index])
        return gym.make(self.reactions[index])

    def load_extraction_bench(self, index):
        # this function returns the extraction environment that the agent has selected
        print(self.extractions[index])
        return gym.make(self.extractions[index])

    def load_distillation_bench(self, index):
        # this function returns the distillation environment that the agent has selected
        print(self.distillations[index])
        return gym.make(self.distillations[index])

    def run_bench(self, bench, env_index, vessel_index, agent_index=0, custom_agent=None):
        """
        bench: the bench that is to be run in the lab
        env_index: which environment(experiment) from the specified bench will be used
        vessel_index: the vessel that will be used in the specified bench
        agent_index: the agent that will be used to perform the experiment
        custom_agent: allows for the user to specify a custom agent
        """
        env = None
        agent = None
        total_reward = 0

        if bench == 'reaction':
            # we need to finish updates to reaction class inorder to get the ability to add a new vessel
            if env_index >= len(self.reactions):
                total_reward -= 10
            else:
                env = self.load_reaction_bench(env_index)
                if vessel_index > self.shelf.open_slot:
                    total_reward -= 10
                else:
                    env.update_vessel(self.shelf.get_vessel(vessel_index))
            if agent_index >= len(list(self.react_agents.keys())):
                total_reward -= 10
            else:
                agent_name = list(self.react_agents.keys())[agent_index]
                agent = self.react_agents[agent_name]
        elif bench == 'extraction':
            if env_index >= len(self.extractions):
                total_reward -= 10
            else:
                env = self.load_extraction_bench(env_index)
                if vessel_index > self.shelf.open_slot:
                    total_reward -= 10
                else:
                    env.update_vessel(self.shelf.get_vessel(vessel_index))
            if agent_index >= len(list(self.extract_agents.keys())):
                total_reward -= 10
            else:
                agent_name = list(self.extract_agents.keys())[agent_index]
                agent = self.extract_agents[agent_name]
        elif bench == 'distillation':
            if env_index >= len(self.distillations):
                total_reward -= 10
            else:
                env = self.load_distillation_bench(env_index)
                if vessel_index > self.shelf.open_slot:
                    total_reward -= 10
                else:
                    env.update_vessel(self.shelf.get_vessel(vessel_index))
            if agent_index >= len(list(self.distill_agents.keys())):
                 total_reward -= 10
            else:
                agent_name = list(self.distill_agents.keys())[agent_index]
                agent = self.distill_agents[agent_name]
        elif bench == 'analysis':
            analysis = np.array([])
            if vessel_index > self.shelf.open_slot or env_index >= len(self.analysis):
                total_reward -= 10
            else:
                analysis = self.analysis_bench.analyze(self.shelf.get_vessel(vessel_index), self.analysis[env_index])
            return total_reward, analysis
        else:
            raise KeyError(f'{bench} is not a recognized bench')
        done = total_reward < 0
        if not done:
            if custom_agent:
                agent = custom_agent
            state = env.reset()
            while not done:
                # env.render(mode=self.render_mode)
                action = agent.run_step(env, state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
            rtn_vessel = env.vessels
            self.shelf.return_vessel_to_shelf(vessel=rtn_vessel)
        return total_reward, np.array([])

    def step(self, action: list):
        done = False
        if action[0] == 0:
            # reaction bench
            bench = 'reaction'
        elif action[0] == 1:
            # extraction bench
            bench = 'extraction'
        elif action[0] == 2:
            # distillation bench
            bench = 'distillation'
        elif action[0] == 3:
            # analysis bench
            bench = 'analysis'
        elif action[0] == 4:
            bench = None
            done = True
        else:
            raise EnvironmentError(f'{action[0]} is not a valid environment')
        reward = 0
        analysis = np.array([])
        if not done:
            env_index = action[1]
            vessel_index = action[2]
            agent_index = action[3]
            reward, analysis = self.run_bench(bench, env_index, vessel_index, agent_index)
        return reward, analysis, done

    def reset(self):
        self.shelf.reset()
        return self.shelf.vessels


