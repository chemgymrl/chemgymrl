## Reaction Bench: Lesson 1

#### Part 1:

In this lesson I will be taking you through how our reaction bench environment works and how an RL agent might interact
with the environment.

The reaction bench environment is meant to as it sounds simulate a reaction, in most reaction benches the agent will
have a number of reagents and the ability to play with the environmental conditions of the reaction and through doing
this the agent is trying to maximize the yield of a certain desired material. For the reaction bench we use a reaction 
file which specifies the mechanics of a certain reaction or multiple reactions. For instance the Wurtz reaction is made
up of 6 different reactions and as such is a very complicated reaction which the agent has to try and learn the
mechanisms of the reaction environment it is in. For this lesson we will be using a simplified version of the wurtz
reaction to introduce you to how actions affect the environment.

Firtst let's load up the environment, I highly recommend you look at the source code for the reaction bench and
reaction, it should help provide insight into how this all works. Further the lesson on creating a custom reaction
environment will also help give insight into the reaction mechanics. In order to run the environment simply run: 
`lessons/reaction_bench/reaction_wurtz.py`. When you run this file you should see a prompt like this in your command
line along with the following graph:

![graph](../../sample_figures/tutorial/wurtz_overlap_command_0.png)

![graph](../../sample_figures/tutorial/wurtz_overlap_0.png)

Understanding the graph above is important to understanding how the agent will have to understand the environment.
On the left we can see the absorbance spectra of the materials in our reaction vessel, and on the right we have
a relative scale of a number of important metrics. From left to right we have time passed, temperature, volume (solvent)
, presure, and the quantity of reagents that we have available to use. All of this data is what the RL agent has inorder
for it to try and optimize the reaction pathway. 

The reaction we are using is as follows:

2 1-chlorohexane + 2 Na --> dodecane + 2 NaCl

This reaction is performed in an aqueous state with ethoxyethane as the solvent.

With all that out of the way let's focus our attention to the action space. For this reaction environemnt our action
space is represented by a 6 element vector. 

|              | Temperature | Volume | 1-chlorohexane | 2-chlorohexane | 3-chlorohexane | Na  |
|--------------|-------------|--------|----------------|----------------|----------------|-----|
| Value range: | 0-1         | 0-1    | 0-1            | 0-1            | 0-1            | 0-1 |

As you might have noticed now, the reaction bench environment deals with a continuous action space. So what exactly do
these continuous values represent? For the environmental conditions, in this case Volume and Temperature 0 represents a
decrease in temperature  or volume by dT or dV (specified in the reaction bench), 1/2 represents no change, and
1 represents an increase by dT or dV. For the chemicals, 0 represents adding no amount of that chemical to the reaction
vessel, and 1 represents adding all of the originally available chemical (there is a negative reward if you try to add
more chemical than is available). 

Now that we understand the action space I encourage you to use this reaction environment and to play around with
the environmental conditions and adding the chemicals into the reaction vessel.

#### Part 2:

Here I will provide instructions on how to maximize the return of this reaction environment.

This is fairly simple for this task and have thus provided some script which demonstrates our strategy, and I encourage
you to try your own strategy and see how it performs. In this case cour strategy is at step 1 to increase the temperature,
keep the volume of solvent constant, and to add all our reagents, in this case 1-chlorohexane and Na. This gives us an
action vector of:

| Temperature | Volume | 1-chlorohexane | 2-chlorohexane | 3-chlorohexane | Na  |
|-------------|--------|----------------|----------------|----------------|-----|
| 1         | 1/2    | 1            | 0            | 0            | 1 |

Then at every next step we are going to keep the solvent volume constant and increase the temperature

| Temperature | Volume | 1-chlorohexane | 2-chlorohexane | 3-chlorohexane | Na  |
|-------------|--------|----------------|----------------|----------------|-----|
| 1         | 1/2    | 0            | 0            | 0            | 0 |

to see this strategy in action simply run `lessons/reaction_bench/reaction_wurtz_automated.py`
you should see a graph like this at the end and recieve a reward of the following:

![final graph]()

![final reward]()

and now we're done! I hope you have a better sense of how the reaction environment works and the process through which
an RL agent must go through to learn the environment.