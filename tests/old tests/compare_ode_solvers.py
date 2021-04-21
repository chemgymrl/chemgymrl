"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
sys.path.append('../../../')
sys.path.append('../../chemistrylab/reactions')
import gym
import chemistrylab
import datetime as dt
import numpy as np
from gym import envs
from chemistrylab.reaction_bench.reaction_bench_v1 import ReactionBenchEnv_0

all_envs = envs.registry.all()
render_mode = "human"

solvers = {'newton', 'RK45', 'RK23', 'DOP853', 'BDF', 'LSODA'}
avg_reward = {}
avg_runtime = {}
total_runs = 1
for i in range(total_runs):
    for solver in solvers:
        print(solver)
        env = ReactionBenchEnv_0()
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
            state, reward, done, _ = env.step(action)
            total_reward += reward
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
