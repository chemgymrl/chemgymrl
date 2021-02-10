## Distillation Bench: Lesson 1

### Familiarizing Actions for the Distillation Bench

In this tutorial, I will teach you how the distillation environment works. In this distillation we are taking the 
`tests/test_extract_vessel.pickle` which is a pickle file generated when we ran the extraction bench. 

It is important to
know that before running the distillation bench, you must first complete both reaction and extraction, in order to follow
the workflow our environment. To find out more, take a look at the [readme.md](https://github.com/CLEANit/chemistrygym) file on our github.

Start by running `/tests/demo_lesson_1d.py` and you should see a series of graphs appear:

![graph](../sample_figures/lesson_1d_image0.PNG)

These graphs show the contents of our containers. There are 3 main containers:
- boiling vessel
- beaker 1
- beaker 2

The boiling vessel contains all the materials at the initial state of the experiment. Beaker 1 can be thought of as a 
condensation vessel which is connected to the distillation vessel via a tube and this will contain all the materials 
that are being boiled off. Beaker 2 is then the storage vessel, where the condensation vessel can be emptied, in order 
to make room for other material.

Now after we load in our desired environment, we should be prompted with a message asking us to pick our desired action and 
the multiplier.

![prompt](../sample_figures/lesson_1d_image1.PNG)

Let me explain what this means by taking a look at our action space. 

Our action space is given by a matrix of shape {0, 1}<sup>n_action x multiplier</sup>. The agent at each time step 
essentially picks from a set of actions, and then a multiplier for that given action. The multiplier affects each action
differently. For example the way the agents chosen multiplier affects heat change is by the following code:

![heatchange](../sample_figures/lesson_1d_image2.PNG)

In this case the heat change is affected by both the user's chosen multiplier and the dQ which is an attribute for the 
maximum change in thermal energy. 

Typically, an agent will choose actions based on what will give a higher reward, and higher rewards are given by getting
a high molar amount of the desired material, in this case, dodecane, as well as a high purity in one particular vessel.

From a chemistry point of view, what's happening is that the agent is extracting a material from the boiling vessel, 
which contains multiple materials, and it does this by utilizing the different material's boiling points. The experiment
begins by raising the temperature of the boiling vessel which will allow certain materials in that vessel to boil off 
into the condensation vessel/beaker 1. The amount of energy required to raise the temperature of the materials in the 
vessel are dependent on their specific heat properties, which can be found in the `chemistrylab/chem_algorithms/material.py`
file. 

As more heat is added we will eventually reach the lowest boiling point of the materials in the vessel, and any more 
heat added will act to evaporate this material. The now gaseous material will rise out of the boiling vessel into the 
tube that feeds into the condensation vessel where it will condense back into its liquid form. In this virtual experiment 
it is assumed that this takes place instantaneously. The amount of material evaporated is dependent on the enthalpy of 
vapour of material being evaporated. 

Once the entirety of the material has been boiled off, the condensation vessel is drained into the storage vessel. Now 
the condensation vessel is empty, the boiling vessel's temperature can then be raised more until the next lowest boiling 
point is reached, thus repeating the process.

The process is repeated until the desired material has been completely evaporated from the boiling vessel into 
condensation vessel. From this point on the desired material is completely isolated and we obtain a hopefully pure sample.
The agent can then choose action 5, or done, in order to end the experiment.

![distillation-diagram](../sample_figures/lesson_1d_image3.PNG)

In lesson 3 in these sets of tutorial for the distillation bench, we will try to get a high reward by obtaining a high 
molar amount of pure dodecane in our condensation vessel. In this tutorial though, we will go ahead and show you the 
basic actions that your agent can choose to do. 

For now, we will manually input the action and the multiplier at each time step, however the whole purpose of these 
environments is to train the agent with reinforcement learning. Hopefully after finishing these lessons, you will give
RL with these agents a go!

Let's get started with the different actions!

**First let's start by adding temperature to the boiling vessel.** This can be acheived by setting the action to 0, and 
making the multiplier an integer higher than 5. Note that a multiplier of 5 will not affect the temperature and any 
lower will decrease temperature.

![add-temp](../sample_figures/lesson_1d_image4.PNG)

Notice how the boiling vessel's temperature is now 374.6 Kelvin. The boiling point of H2O as specified in the material.py
file is 373.15 K. So we are now exceeding this temperature which allows us to boil off water, in this case 0.276 mol of
it. 

We can see this boiled off H2O in our beaker_0 in the diagrams which you'll notice before was empty.

![add-temp](../sample_figures/lesson_1d_image5.PNG)

**After, this we can test out pouring from the condensation vessel into the storage vessel.**

![add-temp](../sample_figures/lesson_1d_image6.PNG)

We can then see that storage vessel is now filled with the H2O poured from the condensation vessel. 

**Note there is a problem with the diagrams right now that should be fixed.**

![pour-beaker1](../sample_figures/lesson_1d_image7.PNG)

**Next, let's increase the temperature one more time, and then test actions 1 and 3.**

First Let's pour from the condensation vessel to boiling vessel after boiling some more materials.

![add-temp](../sample_figures/lesson_1d_image8.PNG)

Noticed that we now have boiled 0.17 mols 2-chlorohexane.

![add-temp](../sample_figures/lesson_1d_image9.PNG)

**The graph should no longer have H2O in beaker_1**

Let's try pouring everything back to the boiling vessel

![pour-to-bv](../sample_figures/lesson_1d_image10.PNG)

You should see now that everything from the condensation vessel is back in the boiling vessel.

**---Should insert graph here but it doesn't show the action correctly---*

Now let's try pouring everything from the boiling vessel into the condensation vessel.

![pour-to-b1](../sample_figures/lesson_1d_image11.PNG)

Notice now that all the materials are in the condensation vessel.

![pour-to-b1](../sample_figures/lesson_1d_image12.PNG)

Next we can test pouring the H2O in the storage vessel into the boiling vessel which is the last action we can select.

![pour-to-bv](../sample_figures/lesson_1d_image13.PNG)

Now you can see that the H2O which was previously in the storage vessel, is now in the boiling vessel.

![pour-to-bv](../sample_figures/lesson_1d_image14.PNG)

We can now then tell the agent to stop the experiment using action 5 and multiplier 0.

![done](../sample_figures/lesson_1d_image15.PNG)

This concludes the end of our tutorial. Hopefully you got a chance to see how the basic actions in the distillation 
environment works and see how you can use the agent in RL applications to maximize the distillation of a desired material.

In the next tutorial we will perform 2 experiments where we heat up the boiling vessel to the maximum temperature it can
reach, and vice versa where we cool it to the lowest temperature.
