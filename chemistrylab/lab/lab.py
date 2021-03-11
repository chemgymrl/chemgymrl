import sys
sys.path.append("../../") # to access `chemistrylab`
sys.path.append('../../chemistrylab/reactions')
from chemistrylab.lab.agent import RandomAgent
from chemistrylab.lab.shelf import Shelf
import gym
import chemistrylab
import numpy as np
from gym import envs


class Lab:
    def __init__(self, render_mode=None):
        all_envs = envs.registry.all()
        self.reactions = [env_spec.id for env_spec in all_envs if 'React' in env_spec.id]
        self.extractions = [env_spec.id for env_spec in all_envs if 'Extract' in env_spec.id]
        self.distillations = [env_spec.id for env_spec in all_envs if 'Distill' in env_spec.id]
        self.react_agents = {'random': RandomAgent()}
        self.extract_agents = {'random': RandomAgent()}
        self.distill_agents = {'random': RandomAgent()}
        self.render_mode = render_mode
        self.shelf = Shelf(max_num_vessels=100)


    def load_reaction_bench(self, index):
        return gym.make(self.reactions[index])

    def load_extraction_bench(self, index):
        return gym.make(self.extractions[index])

    def load_distillation_bench(self, index):
        return gym.make(self.distillations[index])

    def run_bench(self, bench, env_index, vessel_index, agent_name='random'):
        env = None
        agent = None
        if bench == 'reaction':
            # we need to finish updates to reaction class inorder to get the ability to add a new vessel
            env = self.load_reaction_bench(env_index)
            env.update_vessel(self.shelf.get_vessel(vessel_index))
            agent = self.react_agents[agent_name]
        elif bench == 'extraction':
            env = self.load_extraction_bench(env_index)
            env.update_vessel(self.shelf.get_vessel(vessel_index))
            agent = self.extract_agents[agent_name]
        elif bench == 'distillation':
            env = self.load_distillation_bench(env_index)
            env.update_vessel(self.shelf.get_vessel(vessel_index))
            agent = self.distill_agents[agent_name]
        else:
            raise KeyError(f'{bench} is not a recognized bench')
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            # env.render(mode=self.render_mode)
            action = agent.run_step(env, state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rtn_vessel = env.vessels
        self.shelf.return_vessel_to_shelf(vessel=rtn_vessel)

