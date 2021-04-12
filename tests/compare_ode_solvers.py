import sys
sys.path.append('../../')
sys.path.append('../../chemistrylab/reactions')
import gym
import chemistrylab
import datetime as dt
import numpy as np
from gym import envs
from chemistrylab.reaction_bench.reaction_bench_v1 import ReactionBenchEnv_0

all_envs = envs.registry.all()
env = ReactionBenchEnv_0()
render_mode = "human"

solvers = {'newton', 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'}
avg_reward = {}
avg_runtime = {}
total_runs = 100
for i in range(total_runs):
    if not(i % 10):
        print(i)
    for solver in solvers:
        env.reaction.solver = solver
        done = False
        state = env.reset()
        total_reward = 0
        round = 0
        start = dt.datetime.now()
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
                end = dt.datetime.now()
                done = True
                if solver not in avg_runtime:
                    avg_runtime[solver] = (end - start) / total_runs
                else:
                    avg_runtime[solver] += (end - start) / total_runs

                if solver not in avg_reward:
                    avg_reward[solver] = total_reward / total_runs
                else:
                    avg_reward[solver] += total_reward / total_runs
            round += 1

for solver in solvers:
    print(solver)
    print(f'avg reward: {avg_reward[solver]}')
    print(f'avg time: {avg_runtime[solver]}')
