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

# show all environments for reaction bench
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'React' in env_spec.id]
print(env_ids)

# trying to get high reward for wurtz reaction of form:
# A-X + A-X --> A-A + X-X and A-X + B-X --> A-B + X-X
# Rewards comes from producing A-A or B-B
# Cannot come from A-B as this doesn't make the desired property
# Desired material in this case is initialized to be 4,5-diethyloctane
# initializes environment
env = gym.make("WurtzReact-v2")
render_mode = "human"

done = False
__ = env.reset()

# shows # of actions available
print('# of actions available: ',env.action_space.shape[0])
num_actions_available = env.action_space.shape[0]

total_steps=0
total_reward=0

reward_over_time=[]
steps_over_time=[]
reactant_1 = []
reactant_2 = []
total_reward_over_time = []

action = np.ones(env.action_space.shape)

while not done:
    # Actions:
    #   a[0] changes the temperature between -dT (a[0] = 0.0) and +dT (a[0] = 1.0)
    #   a[1] changes the Volume between -dV (a[1] = 0.0) and +dV (a[1] = 1.0)
    #   a[2:] adds between none (a[2:] = 0.0) and all (a[2:] = 1.0) of each reactant
    if total_steps  < 20:
        action[0] = 1
        action[1] = 1
        action[2] = 1    # 1-chlorohexane
        action[3] = 1    # 2-chlorohexane
        action[4] = 1    # 3-chlorohexane
        action[5] = 1    # Na

        '''
        # Adding Reactants not needed:
        action[0] = 1
        action[1] = 1
        action[5] = 1
        action[4] = 1
        action[2] = 1
        action[3] = 1
        '''

    # perform the action and update the reward
    state, reward, done, __ = env.step(action)
    print('-----------------------------------------')
    print('total_steps: ', total_steps)
    print('reward: %.2f ' % reward)
    total_reward += reward
    print('total reward: %.2f' % total_reward)
    # print(state)

    # render the plot
    env.render(mode=render_mode)
    # sleep(2)

    # increment one step
    total_steps += 1

    # append arrays for states over time
    reward_over_time.append(reward)
    total_reward_over_time.append(total_reward)
    steps_over_time.append(total_steps)
    reactant_1.append(env.state[6])
    reactant_2.append(env.state[7])

# ask user if they want to see stats of reaction vessel
show_stats = input("Show Reaction Vessel Stats ('Y'/'N') >>> ")

if show_stats.lower() in ["y", "yes"]:
    # open and check the material dict
    vessel_path = os.path.join(os.getcwd(), "reaction_vessel.pickle")
    with open(vessel_path, 'rb') as open_file:
        v = pickle.load(open_file)

    print("")
    print("---------- VESSEL ----------")
    print("Label: {}".format(v.label))

    print("")
    print("---------- THERMODYNAMIC VARIABLES ----------")
    print("Temperature (in K): {:e}".format(v.temperature))
    print("Volume (in L): {:e}".format(v.volume))
    print("Pressure (in kPa): {:e}".format(v.pressure))

    print("")
    print("---------- MATERIAL_DICT ----------")
    for material, value_list in v._material_dict.items():
        print("{} : {}".format(material, value_list))

    print("")
    print("---------- SOLUTE_DICT ----------")
    for solute, value_list in v._solute_dict.items():
        print("{} : {}".format(solute, value_list))

reward_over_time = pd.Series(reward_over_time)
steps_over_time = pd.Series(steps_over_time)

reward_over_time = pd.Series(reward_over_time)
total_reward_over_time = pd.Series(total_reward_over_time)
steps_over_time = pd.Series(steps_over_time)
reactant_1 = pd.Series(reactant_1)
reactant_2 = pd.Series(reactant_2)

# graph states over time
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.plot(steps_over_time, reactant_1)
ax1.set_title('Steps vs. Reactant 1 (3-chlorohexane)')
ax1.set_xlabel('Steps')
ax1.set_ylabel('3-chlorohexane')

ax2.plot(steps_over_time, reactant_2, 'tab:orange')
ax2.set_title('Steps vs. Reactant 2 (Na)')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Na')

ax3.plot(steps_over_time, reward_over_time, 'tab:green')
ax3.set_title('Steps vs Reward')
ax3.set_xlabel('Steps')
ax3.set_ylabel('Reward')

ax4.plot(steps_over_time, total_reward_over_time, 'tab:red')
ax4.set_title('Steps vs Total Reward')
ax4.set_xlabel('Steps')
ax4.set_ylabel('Total Reward')

fig.tight_layout()
plt.savefig('Final Subplots Demo Lesson 3.png')
plt.show()