'''
A demo displaying ExtractWorld's utility in creating environments and performing experiments.

:title: demo_ExtractWorld.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-06-24
'''

import gym
import numpy as np
import sys
from time import sleep
import time

sys.path.append("../") # to access chemistrylab
import chemistrylab

# create a registered environment
env = gym.make('ExtractWorld_0-v1')
env.reset()

# render the initial state
env.render()

# perform a sample action and render the resulting state
action = np.array([3, 2])
state, reward, done, __ = env.step(action)
env.render()

__ = input('Press ENTER to continue')

done = False
step_num = 0
total_reward = 0.0

while not done:
    # Select a random action
    action_space = env.action_space
    action = action_space.sample()

    # perform the random action
    state, reward, done, __ = env.step(action)
    
    total_reward += reward
    env.render()
    sleep(3)
    step_num += 1
