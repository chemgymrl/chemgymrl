'''
ChemistryGym Demo

:title: demo_chemistrygym.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-07-03
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

# -------------------- # DISTILLATION BENCH DEMO # -------------------- #
__ = input('PRESS ENTER TO START DISTILLATION BENCH.')

# create a registered environment
d_env = gym.make('Distillation-v0')
d_env.reset()

# queue and perform the Extraction Vessel's pour by volume action with a multiplier of 0.5
action = np.array([0, 10])
__, __, __, __ = d_env.step(action)

__ = input('PRESS ENTER TO CONTINUE DISTILLATION BENCH')

done = False
step_num = 0
total_reward = 0.0

while not done:
    # select and perform a random action
    # actions consist of arrays of two elements
    #   action[0] is a number indicating the event to take place
    #   action[1] is a number representing a multiplier for the event
    # Actions and multipliers are included below:
    #   0: Valve (Speed multiplier, relative to max_valve_speed)
    #   1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
    #   2: Pour B1 into ExV (Volume multiplier, relative to max_vessel_volume)
    #   3: Pour B2 into ExV (Volume multiplier, relative to max_vessel_volume)
    #   4: Pour ExV into B2 (Volume multiplier, relative to default vessel volume)
    #   5: Pour S1 into ExV (Volume multiplier, relative to max_vessel_volume)
    #   6: Pour S2 into ExV (Volume multiplier, relative to max_vessel_volume)
    #   7: Done (Multiplier doesn't matter)
    action_space = d_env.action_space
    action = action_space.sample()

    # ensure atleast 5 steps are completed before the done exit action
    if step_num < 5:
        while action[0] == 5:
            action_space = d_env.action_space
            action = action_space.sample()

    # perform the random action and update the reward
    state, reward, done, __ = d_env.step(action)
    total_reward += reward

    # render each of the three plots
    d_env.render()
    sleep(2)

    step_num += 1

__ = input("PRESS ENTER TO EXIT")