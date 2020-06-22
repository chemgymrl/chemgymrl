import gym
import numpy as np
import pickle
from time import sleep

# ensure all necessary modules can be found
import sys
sys.path.append("../") # to access chemistrylab
sys.path.append("../chemistrylab/reactions")
import chemistrylab

# Demo for using ODEWorld-v0, Custom Reaction

perfect = False # Use a perfect policy

# Initialize the environment
env = gym.make('ODEWorld_0-v0')
render_mode = "human"

# Reset the environment to get initial state
# State [0] is seconds since the start of reaction
# State [1] is the thermostat temperature of the system
# State [2 and beyond] is the remaining amounts of each reactant
s = env.reset()
env.render(mode=render_mode)
wait = input('PRESS ENTER TO CONTINUE.')

done = False
i = 0

# Game will play until: 20 steps are reached
total_reward = 0.0

while not done:
    # Select random actions
    # Action [0] changes the temperature between -dT (a[0]=0.0) and +dT (a[1]=1.0)
    # Action [1] changes the Volume between -dV and +dV
    # Action [2] adds reactant A to the system between 0 mol (a[1]=0.0) and 1 mol (a[1]=1.0)
    # Action [3] adds reactant B to the system between 0 mol (a[2]=0.0) and 1 mol (a[2]=1.0)
    if perfect:
        if i == 0:
            action = np.ones(env.action_space.shape)
            action[1] = 0.0
        else:
            action = np.zeros(env.action_space.shape)
            action[0] = 1.0
    else:
        action = env.action_space.sample()

    state, reward, done, __ = env.step(action)
    print(env.vessels._material_dict)
    total_reward += reward

    env.render(mode=render_mode)
    sleep(1) # Wait before continuing

    i += 1
#    print("\nCompleted Step {}".format(i))
#    print("Generated {} Reward".format(reward))
#    print("Completed All Steps? {}".format(done))

#print("\nExecuted {} Actions.".format(i))
#print("Generated {} Mol of Reward".format(round(total_reward, 3)))

#wait = input('PRESS ENTER TO EXIT.')

with open('vessel_experiment_0.pickle', 'rb') as handle:
    b = pickle.load(handle)
    print(b._material_dict)
