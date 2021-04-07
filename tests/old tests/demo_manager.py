'''
LabManager Demo

:title: demo_manager.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-08-07
'''

# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=unused-import
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

# import all the required external modules
import gym
import numpy as np
import os
import pickle
import sys
from time import sleep

# ensure all necessary modules can be found
sys.path.append("../") # to access chemistrylab
sys.path.append("../chemistrylab/reactions") # to access all reactions

# import all local modules
import chemistrylab

# -------------------- # REACTION BENCH DEMO # -------------------- #
__ = input("PRESS ENTER TO START LAB MANAGER.")

# Initialize the environment
m_env = gym.make('LabManager-v0')
render_mode = "human"

# Reset each environment to get all initial states
__ = m_env.reset()

__ = input('PRESS ENTER TO CONTINUE LAB MANAGER.')

done = False
i = 0

# Game will play until: a certain number of steps are completed
total_reward = 0.0

while not done:
    # get the complete environment space to select an environment from;
    # `env_space` is a number indicating the index of the selected environment
    # in the list outputted by `manager_v1.py`'s `get_environments` function
    env_space = m_env.env_space

    # select an environment from the environment space
    env_index = env_space.sample()[0]

    # set the selected environment in the lab manager engine
    m_env.set_environment(env_index)

    # get an array containing the action spaces of each available environment
    action_spaces = m_env.action_space_array

    # get the action spaces for the selected environment
    action_space = action_spaces[env_index]

    # select an action from the action space
    action = action_space.sample()

    # perform the action and update the reward
    state, reward, done, __ = m_env.step(action)
    total_reward += reward

    # render the plot and wait before continuing
    # m_env.render(env_index, mode=render_mode)
    # sleep(2)

    i += 1