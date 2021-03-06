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
# import all the required external modules
import gym
import numpy as np
import os
import pickle
import sys
from time import sleep
from gym import envs
import matplotlib.pyplot as plt
import pandas as pd

# ensure all necessary modules can be found
sys.path.append("../../../tests/") # to access chemistrylab
sys.path.append("../chemistrylab/reactions") # to access all reactions

# import all local modules
import chemistrylab

# show all environments for distillation bench
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'Distillation' in env_spec.id]
print(env_ids)

# allows user to pick which environment they want to use
# initializes environment
select_env = int(input(f"Enter a number to choose which environment you want to run (0 - {len(env_ids) - 1}): \n"))
env = gym.make(env_ids[select_env])
render_mode = "human"

done = False
__ = env.reset()
print('\n')

# shows # of actions available
# for distillation bench there are two elements
# action[0] is a number indicating the event to take place
# action[1] is a number representing a multiplier for the event
# Actions and multipliers include:
#   0: Add/Remove Heat (Heat Value multiplier, relative of maximal heat change)
#   1: Pour BV into B1 (Volume multiplier, relative to max_vessel_volume)
#   2: Pour B1 into B2 (Volume multiplier, relative to max_vessel_volume)
#   3: Wait for boil vessel temp to decrease towards room temp (multiplier == 0, wait until room temp == true)
#   4: Done (Value doesn't matter)

action_set = ['Add/Remove Heat', 'Pour BV into B1', 'Pour B1 into B2', 'Done']
assert env.action_space.shape[0] == 2
print(env.action_space)

total_steps=0
total_reward=0

temp = []
steps_over_time = []

while not done:

    action = np.zeros(env.action_space.shape[0])

    if total_steps == 0:
        action = np.array([0,500])
    elif total_steps == 1:
        action = np.array([0,450])
    elif total_steps == 2:
        action = np.array([2,10])
    elif total_steps == 3:
        action = np.array([0,75])
    elif total_steps == 4:
        action = np.array([3,0])

    # perform the action and update the reward
    state, reward, done, __ = env.step(action)
    print('-----------------------------------------')
    print('total_steps: ', total_steps)
    print('reward: %.2f ' % reward)
    total_reward += reward
    print('total reward: %.2f ' % total_reward)
    print('Temperature of boiling vessel: %.1f ' % env.boil_vessel.temperature, ' K \n')
    # print(state)

    # render the plot
    if total_steps == 4:
        input('h')
    env.render(mode=render_mode)
    # sleep(1)

    #increment one step
    total_steps += 1

