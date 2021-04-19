[chemgymrl.com](https://chemgymrl.com/)

## Reaction Bench Lesson 2

#### Getting a high reward in a reaction of form: 

A-X + A-X -> A-A + X-X and A-X + B-X -> A-B + X-X

In this lesson we will try to get a high reward in the reaction of the form above. Rewards will come from producing either A-A or B-B. It's important to note that the reward cannot come from A-B as this doesn't make the desired property. The reaction we will be taking an in depth look at in this lesson is:

2 3-chlorohexane + 2 Na -> 4,5-diethyloctane + 2 NaCl

We will try to get the desired material: 4,5-diethyloctane

In similar fashion to lesson 1, the reactions used in this lesson are found in the available reactions file. This particular lesson will use the reaction file 'chloro_wurtz_v1.py' and is registered under the id 'WurtzReact-v2'

From lesson 1 we know that our action space is a 6 element vector represented by:

|              | Temperature | Volume | 1-chlorohexane | 2 chlorohexane | 3-chlorohexane | Na  | 
|--------------|-------------|--------|----------------|----------------|----------------|-----|
| Value range: | 0-1         | 0-1    | 0-1            | 0-1            | 0-1            |0-1|

Each index corresponds to the following label and how we change them. For example is action[0] = 0 then the temperature will decrease.
If it is set to 0.5 then it will stay the same and if set to 1 then the temperature will increase.

First let's start by importing all the modules we need.


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
%matplotlib inline

# ensure all necessary modules can be found
sys.path.append("../") # to access chemistrylab
sys.path.append("../chemistrylab/reactions") # to access all reactions

# import all local modules
import chemistrylab
```

This will show all the environments we can currently run. Eventually you can create your own environments with different reactions and target material using the reaction template.


```python
# show all environments for reaction bench
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'React' in env_spec.id]
print(env_ids)
```

This explains the simulated reaction we are trying to simulate and is initializing the reaction environment.


```python
# trying to get high reward for wurtz reaction of form:
# A-X + A-X --> A-A + X-X and A-X + B-X --> A-B + X-X
# Rewards comes from producing A-A or B-B
# Cannot come from A-B as this doesn't make the desired property
# Desired material in this case is initialized to be 4,5-diethyloctane
# initializes environment
env = gym.make("WurtzReact-v2")
render_mode = "human"
```


```python
done = False
__ = env.reset()

# shows # of actions available
print('# of actions available: ',env.action_space.shape[0])
num_actions_available = env.action_space.shape[0]

total_steps=0
total_reward=0
```

We will store certain values in these arrays so can plot them later on to visually show how each variable changes over time.


```python
reward_over_time=[]
steps_over_time=[]
reactant_1 = []
reactant_2 = []
total_reward_over_time = []

action = np.ones(env.action_space.shape)
```

The key to achieving a high reward in this simulation is to only add the reactants that are needed for the reaction to 
continue. This means that we will only add 3-chlorohexane and Na with our actions. This will allow us to maximize our 
reward as a large quantity of these reactants means the reaction with our target material will occur more often. We 
do this by running the following commands:

![actions](../tutorial_figures/reaction-lesson-2/actions_reaction.png)

Notice that we're only adding the reactants we need for the reaction to continue; 3-chlorohexane

![reaction](../tutorial_figures/reaction-lesson-2/reaction.png)

<a style="font-size: 5px">(source http://butane.chem.uiuc.edu/pshapley/Enlist/Labs/RateEquilib/RateEquilib.html)</a>

Let's run our program and see what happens!


```python
while not done:
    # Actions:
    #   a[0] changes the temperature between -dT (a[0] = 0.0) and +dT (a[0] = 1.0)
    #   a[1] changes the Volume between -dV (a[1] = 0.0) and +dV (a[1] = 1.0)
    #   a[2:] adds between none (a[2:] = 0.0) and all (a[2:] = 1.0) of each reactant
    if total_steps  < 20:
        action[0] = 1
        action[1] = 1
        action[2] = 0    # 1-chlorohexane
        action[3] = 0    # 2-chlorohexane
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
```

Notice that we get a total reward of 2.45. A visual representation of the reactants being used and total reward increasing can be seen in the subplot we produce!

![subplot](../tutorial_figures/reaction-lesson-2/subplots.png)

This simply shows us the stats of the reaction vessel. It essentially shows everything from thermodynamic variables, to the amount of material


```python
# ask user if they want to see stats of reaction vessel
show_stats = input("Show Reaction Vessel Stats ('Y'/'N') >>> ")

if show_stats.lower() in ["y", "yes"]:
    # open and check the material dict
    vessel_path = os.path.join(os.getcwd(), "vessel_experiment_0.pickle")
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
```

This part of the code plots certain states over time.


```python

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
```

For the second part of the experiment let's uncomment the code that adds the reactants not needed and run our code again.

![code](../tutorial_figures/reaction-lesson-2/actions_for_bad_reward.PNG)

If we run this code we'll notice that our reward is significantly lower. It is now only 1.06 which is a drop-off from our previous set of actions. Once again, the reason this is happening is that other reactions are taking place instead of the reaction that produces our desired material. 

The next step for this reaction environment is to write an RL implementation that will allow the agent to solve this problem for you essentially maximizing our output of the desired material! 
