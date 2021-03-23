import sys
sys.path.append('../../')
sys.path.append('../../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np
from gym import envs
from chemistrylab.reaction_bench.reaction_bench_v1 import ReactionBenchEnv_0

all_envs = envs.registry.all()
env = ReactionBenchEnv_0()
render_mode = "human"

solvers = ['newton', 'RK45']
for solver in solvers:
    print(solver)
    env.reaction.solver = solver
    done = False
    state = env.reset()
    total_reward = 0
    round = 0

    while not done:
        action = np.zeros(env.action_space.shape[0])
        if round == 0:
            action = np.ones(env.action_space.shape[0])
        else:
            action[0] = 1
            action[1] = 1

        # print('--------------------')
        # print(round)
        # for i, a in enumerate(action_set):
        #     action[i] = float(input(f'{a}: '))
        state, reward, done, _ = env.step(action)
        total_reward += reward
        # print(f'reward: {reward}')
        # print(f'total_reward: {total_reward}')
        if round == 20:
            done = True
            print(f'total_reward: {total_reward}')
        round += 1
