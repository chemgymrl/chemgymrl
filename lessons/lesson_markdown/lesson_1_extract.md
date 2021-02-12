## Extraction Bench: Lesson 1
### Using a non-polar solute to extract a solute from water

In this tutorial, we am going to walk you through how our environment works and hopefully give some insight into how an
RL agent might interact with the envirionment. In this extraction we are going to be using oil to extract sodium and
chlorine from water. We are going to be using this file `lessons/extraction_bench/oil_water.py` in order to interact
with the environment and to try and achieve a high reward!

So start by running the `lessons/extraction_bench/oil_water.py` you should see a series of graphs like these appear:

![graph](../../sample_figures/tutorial/oil_and_water_0.png)

These graphs show the contents of each of our containers and the level of seperation between the materials. The graphs
to the right then show the layers of materials forming in the container.

When we start the environment we will see that we have a container filled with water, Na and Cl. The objective of this
environment is to get the Na out of the water, but how might we do this? As the chemists amoung us will say this is a
very simple task, and indeed it is. Using a non-polar solvent we can get the sodium and chlorine to diffuse from the
water into that solvent, in this case we can use oil as our non-polar solvent! so now that the environemnt is loaded,
you should see a prompt in the command line. 

![command line prompt](../../sample_figures/tutorial/oil_water_console_0.png)

Before we start playing around with the environment, let's take a look at
the action space. For this case our action space is a matrix: {0, 1}<sup>n_action x multiplier</sup> in essence at each
time step the RL agent selects from a set of actions and then in turn picks a multiplier for that action. 

| Action:                | Multiplier 0: | Multiplier 1: | Multiplier 2: | Multiplier 3: |
|------------------------|---------------|---------------|---------------|---------------|
| Mix beaker 1           | 0             | 1             | 0             | 0             |
| Drain beaker 1 into 2  | 0             | 0             | 0             | 0             |
| Pour oil into beaker 1 | 0             | 0             | 0             | 0             |

As an example the agent might want to pour a certain amount of solution out of our container so the row controls what action they take,
and the column represents an action multiplier. So in the example of pouring out a certain amount of solution, the
multiplier might control how much of the solution we pour out of our container. so our action matrix will thus be a
matrix of zeroes with a single 1. In the set up of the environment we select a dT, this dT represents the period of time
we take in between each action. Now that we have a better idea of our action space we can move onto testing out the
environment. As stated earlier we need to use oil to extract the sodium from the water, but if we look at the graph now,
we can clearly see that there is no oil in the container, so let's add some!

![image of command](../../sample_figures/tutorial/oil_water_console_1.png)

![image of graphs with added oil](../../sample_figures/tutorial/oil_and_water_1.png)

Now that we've added the oil we need to mix the vessel to get the sodium to transfer into the oil, so let's mix the
vessel!

![image of command](../../sample_figures/tutorial/oil_water_console_2.png)

![image of graphs mixed](../../sample_figures/tutorial/oil_and_water_2.png)

Now that we have done some mixing we need to wait for the oil to settle to the top of the water so we can drain the
water. Keep repeating the following command until the graph settles.

![image of command](../../sample_figures/tutorial/oil_water_console_3.png)

![image of graphs waiting ](../../sample_figures/tutorial/oil_and_water_3.png)

Now that the water and oil have settled we want to drain out our water into beaker 1 so that we can pour out our oil
into vessel 2.

![image of command](../../sample_figures/tutorial/oil_water_console_4.png)

![image of graphs draining](../../sample_figures/tutorial/oil_water_4.png)

Keep repeating the command the changing the multiplier as needed until the graph looks something like this:

![image of graphs draining](../../sample_figures/tutorial/oil_water_5.png)

Now we pour the oil into vessel 2.

![image of command](../../sample_figures/tutorial/oil_water_console_5.png)

![image of graphs pouring](../../sample_figures/tutorial/oil_water_6.png)

Now if we want to we can pour back the water from vessel 1 into our extraction vessel and repeat the process to get a
more of the sodium out of the oil. However, for an introduction this much should satisfy, now that we have finished,
we want to see how well we did so now we enter the done command.

![image of command](../../sample_figures/tutorial/oil_water_console_7.png)

We hope this tutorial helped with your understanding of how an agent might interact with the extraction environmenment!