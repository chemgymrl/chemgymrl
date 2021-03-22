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


class Lab(gym.Env, ABC):
    def __init__(self, render_mode=None, max_num_vessels=100):
        all_envs = envs.registry.all()
        self.reactions = [env_spec.id for env_spec in all_envs if 'React' in env_spec.id]
        self.extractions = [env_spec.id for env_spec in all_envs if 'Extract' in env_spec.id]
        self.distillations = [env_spec.id for env_spec in all_envs if 'Distill' in env_spec.id]
        self.react_agents = {'random': RandomAgent()}
        self.extract_agents = {'random': RandomAgent()}
        self.distill_agents = {'random': RandomAgent()}
        self.render_mode = render_mode
        self.max_number_vessels = max_num_vessels
        self.shelf = Shelf(max_num_vessels=max_num_vessels)
        self.action_space = gym.spaces.MultiDiscrete([4,
                                                      max([len(self.reactions),
                                                           len(self.extractions),
                                                           len(self.distillations)]),
                                                      max_num_vessels,
                                                      max([len(self.react_agents),
                                                           len(self.extract_agents),
                                                           len(self.distill_agents)])])
        # have to get back to this part
        self.observation_space = None

    def load_reaction_bench(self, index):
        print(self.reactions[index])
        return gym.make(self.reactions[index])

    def load_extraction_bench(self, index):
        print(self.extractions[index])
        return gym.make(self.extractions[index])

    def load_distillation_bench(self, index):
        print(self.distillations[index])
        return gym.make(self.distillations[index])

    def run_bench(self, bench, env_index, vessel_index, agent_index=0):
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
            return 0
        else:
            raise KeyError(f'{bench} is not a recognized bench')
        done = total_reward < 0
        if not done:
            state = env.reset()
            while not done:
                # env.render(mode=self.render_mode)
                action = agent.run_step(env, state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
            rtn_vessel = env.vessels
            self.shelf.return_vessel_to_shelf(vessel=rtn_vessel)
        return total_reward

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
        if not done:
            env_index = action[1]
            vessel_index = action[2]
            agent_index = action[3]
            reward = self.run_bench(bench, env_index, vessel_index, agent_index)
        return reward, done

    def reset(self):
        self.shelf.reset()
        return self.shelf.vessels
