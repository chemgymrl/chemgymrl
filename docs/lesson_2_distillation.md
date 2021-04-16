# Distillation Bench: Lesson 2

This experiment consists of 3 main parts. We will:
- Increase to the maximum temperature with a multiplier of 125
- Increase to the maximum temperature with a multiplier of 500
- Decrease to the minimum temperature with a multiplier of 1
    
We will see how these actions play out in the distillation bench environment and gain more intuition into how the distillation bench works.

First let's start by importing all the required modules.


```python
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
```


```python
# ensure all necessary modules can be found
sys.path.append("../") # to access chemistrylab
sys.path.append("../chemistrylab/reactions") # to access all reactions
```


```python
# import all local modules
import chemistrylab
```


```python
# show all environments for distillation bench
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'Distillation' in env_spec.id]
print(env_ids)
```

    ['Distillation-v0']
    

You'll notice that the setup is the same as in lesson 1. Nothing changes for the initialization in this project. The main difference is in the loop later.


```python
# allows user to pick which environment they want to use
# initializes environment
select_env = int(input(f"Enter a number to choose which environment you want to run (0 - {len(env_ids) - 1}): \n"))
env = gym.make(env_ids[select_env])
render_mode = "human"
```

    Enter a number to choose which environment you want to run (0 - 0): 
    0
    


```python
done = False
__ = env.reset()
print('\n')
```

    
    
    


```python
# shows # of actions available
# for distillation bench there are two elements
# action[0] is a number indicating the event to take place
# action[1] is a number representing a multiplier for the event
# Actions and multipliers include:
#   0: Pour BV into B1 (Volume multiplier, relative to max_vessel_volume)
#   1: Pour B1 into B2 (Volume multiplier, relative to max_vessel_volume)
#   2: Wait for boil vessel temp to decrease towards room temp (multiplier == 0, wait until room temp == true)
#   3: Done (Value doesn't matter)
#   4: Done (Value doesn't matter)
action_set = ['Add/Remove Heat', 'Pour BV into B1', 'Pour B1 into B2', 'Wait', 'Done']
assert env.action_space.shape[0] == 2
```


```python
total_steps=0
total_reward=0
```

Here we see that there are 3 main actions, where 2 are commented out. Based on which part of the lesson you're on, please uncomment the needed action.


```python
while not done:

    # ACTION 1
    # increase temperature all the way up
    action = np.array([0,125])

    # ACTION 2
    # results in temperature being too high and all material is boiled off in the vessel
    # action = np.array([0,500])

    # ACTION 3
    # decrease temperature all the way down
    # action = np.array([0,0])



    # perform the action and update the reward
    state, reward, done, __ = env.step(action)
    print('-----------------------------------------')
    print('total_steps: ', total_steps)
    print('reward: %.2f ' % reward)
    total_reward += reward
    print('total reward: %.2f' % total_reward)
    print(action)
    print('Temperature of boiling vessel: %.1f ' % env.boil_vessel.temperature, ' K \n')
    # print(state)

    # render the plot
    env.render(mode=render_mode)
    # sleep(1)

    #increment one step
    total_steps += 1
```

## Action 1
#### Increasing the temperature of the boiling vessel 


```
# ACTION 1
# increase temperature all the way up
action = np.array([0,125])
```

When we run this code we should see that for every timestep that there is a heat change of 24000 joules and that the 
temperature eventually reaches 1738.0 Kelvin. We end up with a final graph that looks like this:

![action 1](../tutorial_figures/distillation-lesson-2/increase_temp_slightly.png)

As you can see most the materials are boiled off from the boiling vessel into the condensation vessel aside from 0.1 mols of NaCl.

## Action 2
#### Increasing the temperature with a higher multiplier

```
# ACTION 2
# results in temperature being too high and all material is boiled off 
in the vessel
action = np.array([0,500])

```

For this we will uncomment action 2. Now running this code gives us an interesting result. Run it yourself to see that we will get:

![negative reward](../tutorial_figures/distillation-lesson-2/negative_reward.png)

We get a negative reward as whenever we try the increase the temperature of an empty boiling vessel the environment will return a reward of -1. Since the multiplier we used in this action was greater than the multiplier used in action 1, we were able to boil off every material in the boiling vessel thus causing us to get this negative reward.

![action 2](../tutorial_figures/distillation-lesson-2/increase_temp_drastically.png)

## Action 3
#### Lower the temperature

We will now lower the temperature the absolute minimum it can be and see what happens. 

Running this results in the following graph:

![graph](../tutorial_figures/distillation-lesson-2/decrease_temp.PNG)

As you will notice, nothing really changes. Of course this isn't much of a surprise since by lowering the temperature  none of the materials in the boiling vessel will boil into the condensation vessel (in this case; if you have materials  with lower boiling points this will change). You'll notice that the minimum temperature the boiling vessel will reach is  297 Kelvin.

This concludes a fairly simple tutorial, to once again, get an intuition of how the distillation bench works. Hopefully  you can see the functionality of the environment. In the next lessons we will see how to achieve a high reward by  trying to isolate a target material, in this case, dodecane, into the condensation vessel. 
