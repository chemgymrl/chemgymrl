[chemgymrl.com](https://chemgymrl.com/)

## Distillation Bench: Lesson 1

### Familiarizing Actions for the Distillation Bench

In this tutorial, we will be going through how the distillation environment works. We will be using the boil vessel method in `distillation_bench_v1.py` in order to generate an input boiling vessel for our distillation bench to use.

It is important to know that normally, before running the distillation bench, you must first complete both reaction and extraction in order to follow the workflow our environments. To find out more, take a look at our [documentation](https://chemgymrl.readthedocs.io/en/latest/WhatIsChemGymRL/)

Before we start talking about loading and running the environment, let's first familiarize ourselves with what's actually going on in the experiment.

### Distillation Process Explained

In the distillation environment there are 3 main containers or vessels.

- boiling vessel (BV)
- beaker 1       (B1)
- beaker 2       (B2)

The boiling vessel (BV) contains all the materials at the initial state of the experiment. Beaker 1 (B1) can be thought of as a  condensation vessel which is connected to the distillation vessel via a tube and this will contain all the materials that are being boiled off. Beaker 2 (B2) is then the storage vessel, where the condensation vessel can be emptied, in order to make room for other material.

![vessels](../tutorial_figures/distillation-lesson-1/vessels_image.png)

<a style="font-size: 10px">(source: https://play.google.com/store/apps/details?id=com.ionicframework.labnursing502994&hl=en_IE&gl=US)</a>

The point of the process is to extract a target material from the boiling vessel, which contains numerous materials, and we do this by utilizing the different material's boiling points. Typically the process begins by raising the temperature of the BV which allows certain materials in that vessel to boil off into the condensation vessel or B1. 

![boiling vessel](../tutorial_figures/distillation-lesson-1/boiling_vessel.png)

<a style="font-size: 10px">(source: https://webstockreview.net/image/experiment-clipart-thermochemistry/1029802.html)</a>

As a material's boiling point is reached, any more temperature added from this point will act to evaporate it. The now gaseous material will rise out of the boiling vessel into the tube that feeds into the condensation vessel where it will condense back into its liquid form. In this virtual experiment  it is assumed that this takes place instantaneously. The amount of material evaporated is dependent on the enthalpy of vapour of material being evaporated.

![distillation process](../tutorial_figures/distillation-lesson-1/distillation_process.png)

<a style="font-size: 10px">(source: https://www.youtube.com/watch?v=Vz2la3947I0)</a>

Once the entirety of the material has been boiled off, the condensation vessel is drained into the storage vessel. Now 
the condensation vessel is empty, the boiling vessel's temperature can then be raised more until the next lowest boiling point is reached, thus repeating the process.

![evaporation](../tutorial_figures/distillation-lesson-1/evaporation.png)

<a style="font-size: 10px">(source: Encyclopedia Britannica, inc.)</a>

The process is repeated until the desired material has been completely evaporated from the boiling vessel into  condensation vessel. From this point on the desired material is completely isolated and we obtain a hopefully pure sample. We can then choose to end the experiment.

In [lesson 3](https://chemgymrl.readthedocs.io/en/latest/lesson_3_distillation/) in these sets of tutorial for the distillation bench, we will try to get a high reward by obtaining a high molar amount of pure dodecane in our condensation vessel. 

For this tutorial, we will just familiarize ourselves with the basic actions, fundamental theory behind distillation, and how you can run the environment on your own!

### Running the environment

We will first start by importing the necessary required modules, both external and local. By now this step should seem very familiar as we have done them in both reaction and extraction lessons.


```python
%matplotlib inline
```


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
sys.path.append('../../')
sys.path.append("../chemistrylab/reactions") # to access all reactions
```


```python
# import local modules
import chemistrylab 
```

We can see all the possible variations of the distillation bench environment which can vary depending on the input vessel (in this case the `test_extract_vessel.pickle`), which is loaded into the boil vessel, as well as the target material. In this and following tutorials our target material will be dodecane


```python
# show all environments for distillation bench
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'Distillation' in env_spec.id]
print(env_ids)
```

We then get prompted with a message asking us to choose the environment we want to run. This is based off the indexing in the environment array we saw from the last cell.


```python
# allows user to pick which environment they want to use
# initializes environment
select_env = int(input(f"Enter a number to choose which environment you want to run (0 - {len(env_ids) - 1}): \n"))
env = gym.make(env_ids[select_env])
render_mode = "human" #select how graphs are rendered
```

We initialize done to False so our agent can run the experiment. We run reset() to return an initial observation.


```python
done = False
__ = env.reset()
print('\n')
```

Here we have the different possible actions that we can take with the environment. The **action_set is an array indexed correspondingly to the action we want to perform.**

The action_space is a multidiscrete action space of shape [4 10].

**The first index allows us to choose from the action set. The second index allows us to pick a multiplier that will affect the action variably depending on our chosen multiplier.**

For example, the following pair of numbers will add a great amount of heat compared to a multiplier of 6. 

Action: 0

Action Multiplier: 10


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

action_set = ['Add/Remove Heat', 'Pour BV into B1', 'Pour B1 into B2', 'Wait','Done']
assert env.action_space.shape[0] == 2

total_steps=0
total_reward=0
```

Note that the multiplier affects each action differently. For examply the way the agents chosen multiplier affects heat change is given by the following code:

![heatchange](../tutorial_figures/distillation-lesson-1/heat_change.PNG)

Also note that when we are performing heat changes, it heavily relies on the given value of dQ. For our lessons we will be using a dQ of 1000.0. Please make sure to change your dQ value to 1000.0 if you are following this lesson to ensure our results stay the same. You can change this value in the `distillation_bench_v1.py` file under the distillation bench folder.

![dQ value](../tutorial_figures/distillation-lesson-1/dQ_value.png)

Typically an agent will choose actions based on what will give a higher reward, and higher reward is given by getting a high molar amount and concentration of the desired material (in our case dodecane) in a particular vessel.

Please input the following action and multipliers:

| Step   | Action   | Multiplier  |
| ------ |:--------:| -----:      |
| 0      | 0        | 350         |
| 1      | 2        | 10          |
| 2      | 0        | 100         |
| 3      | 2        | 10          |
| 4      | 1        | 10          |
| 5      | 4        | 0           |


```python
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
    
    #increment one step
    total_steps += 1
```

#### Step 0: Adding temperature to the vessel

- action: 0
- multiplier: 350

This will result in a temperature reaching the boiling point of water, which you will notice is now boiled off in beaker_0 (or the condensation vessel)

![add-temp](../tutorial_figures/distillation-lesson-1/boil_water.png)

#### Step 1: Pour from condensation to storage vessel

- action: 2
- multiplier: 10

We can then see that storage vessel is now filled with the H2O poured from the condensation vessel. 

![pour-to-beaker1](../tutorial_figures/distillation-lesson-1/pour-to-beaker1.png)

#### Step 2: Add some more temperature

- action: 0
- multiplier: 100

We can now add more temperature in order to boil off 2-chlorohexane into the now empty condensation vessel. 2-chlorohexane is boiled off as in the remaining materials in the boiling vessel, it has the lowest boiling point.

![boil 2-chlorohexane](../tutorial_figures/distillation-lesson-1/boil_2-chl.png)

#### Step 3: Pouring again from condensation to storage

- action: 2
- multiplier: 10

We can again pour the contents of the condensation vessel to the storage vessel

![pour 2-chlorohexane](../tutorial_figures/distillation-lesson-1/pour_2-chl.png)

#### Step 4: Pour everything from boiling vessel into condensation vessel

- action: 1
- multiplier: 10

Notice now that all the materials are in the condensation vessel.

![pour boiling vessel](../tutorial_figures/distillation-lesson-1/pour-bv.png)

#### Step 5: Ending the experiment

- action: 4
- multiplier: 0

### End of the lesson

This concludes the end of our tutorial. Hopefully you got a chance to see how the basic actions in the distillation environment works and see how you can use the agent in RL applications to maximize the distillation of a desired material.

In the next tutorial we will try to perform the distillation process and try to get a high reward by isolating dodecane.
