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
#   3: Pour B1 into BV (Volume multiplier, relative to max_vessel_volume)
#   4: Pour B2 into BV (Volume multiplier, relative to max_vessel_volume)
#   5: Done (Value doesn't matter)

action_set = ['Add/Remove Heat', 'Pour BV into B1', 'Pour B1 into B2', 'Pour B1 into BV', 'Pour B2 into BV', 'Done']
assert env.action_space.shape[0] == 2
print(env.action_space)

total_steps=0
total_reward=0

temp = []
steps_over_time = []

while not done:

    action = np.zeros(env.action_space.shape[0])

    for index, action_desc in enumerate(action_set):
        print(f'{index}: {action_desc}')
    print('Please enter an action and an action multiplier')
    for i in range(2):
        message = 'Action'
        if i == 1:
            message = 'Action Multiplier:'
        action[i] = int(input(f'{message}: '))


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
    env.render(mode=render_mode)
    # sleep(1)

    #temp.append(env.state[1])
    steps_over_time.append(total_steps)

    #increment one step
    total_steps += 1

'''
temp = pd.Series(temp)
steps_over_time = pd.Series(steps_over_time)

plt.plot(steps_over_time,temp)
plt.ylabel('temperature')
plt.xlabel('steps')
plt.savefig('steps vs temperature')
plt.tight_layout()
plt.savefig('Distilattion Lesson 1 Graph')
plt.show()
'''