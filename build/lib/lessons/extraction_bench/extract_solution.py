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
sys.path.append('../../')
sys.path.append('../../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np
from gym import envs
from time import sleep

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'Extract' in env_spec.id]
print(env_ids)
select_env = int(input(f"enter a number to chose which environment you want to run (0-{len(env_ids) - 1}): "))
env = gym.make(env_ids[select_env])
render_mode = "human"

assert 2 == env.action_space.shape[0]

done = False
state = env.reset()
total_reward = 0
reward = 0
while not done:
    env.render(mode=render_mode)
    sleep(1)
    # action = np.zeros(env.action_space.shape[0])
    print(f'reward: {reward}')
    print(f'total_reward: {total_reward}')
    print('--------------------')
    action = env.action_space.sample()
    print(action)
    print('--------------------')
    # for i in range(2):
    #     action[i] = int(input(f'{i}: '))
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        wait = input("PRESS ENTER TO QUIT")
