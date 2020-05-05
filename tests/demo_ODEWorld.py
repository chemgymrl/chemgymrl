import gym
import chemistrylab
import numpy as np
from time import sleep

# Demo for using ODEWorld-v0, Reaction 0

perfect = False # Use a perfect policy

# Initialize the environment
env = gym.make('ODEWorld_0-v0')
render_mode = 'human'

# Reset the environment to get initial state
# State [0] is seconds since the start of reaction
# State [1] is the thermostat temperature of the system
# State [2] is the remaining amount of A
# State [3] is the remaining amount of B
s = env.reset()
env.render(mode = render_mode)
wait = input('PRESS ENTER TO CONTINUE.')

d = False
i = 0

# Game will play until: 20 steps are reached
reward = 0.0
while not d:
    # Select random actions
    # Action [0] changes the temperature between -50 K (a[0]=0.0) and +50 K (a[1]=1.0)
    # Action [1] changes the Volume between -dV and +dV
    # Action [2] adds reactant A to the system between 0 mol (a[1]=0.0) and 1 mol (a[1]=1.0)
    # Action [3] adds reactant B to the system between 0 mol (a[2]=0.0) and 1 mol (a[2]=1.0)
    if perfect:
        if i == 0:
            a = np.ones(env.action_space.shape)
            a[1] = 0.0
        else:
            a = np.zeros(env.action_space.shape)
            a[0] = 1.0
    else:
        a = env.action_space.sample()
    s, r, d, _ = env.step(a)
    reward += r
    env.render(mode = render_mode)
    sleep(1) # Wait 1 seconds before continuing
    i += 1
    print(i, r, d)

print('Total Return = %.3f' % (reward))
wait = input('PRESS ENTER TO EXIT.')

