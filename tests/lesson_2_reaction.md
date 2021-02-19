## Reaction Bench: Lesson 2

### Getting a high reward in a reaction of form: 
### A-X + A-X --> A-A + X-X and A-X + B-X --> A-B + X-X

In this lesson we will try to get a high reward in the reaction of the form above. Rewards will come from producing 
either A-A or B-B. It's important to note that the reward cannot come from A-B as this doesn't make the desired 
property. The reaction we will be simulating in this lesson is specifically:
- 2 3-chlorohexane + 2 Na --> 4,5-diethyloctane + 2 NaCl

In similar fashion to lesson 1, the reactions used in this lesson are found in the reaction file. The reaction above is 
1 of the 6 total reactions in the Wurtz reaction.

After reading lesson 1 we should be familiar with loading and initialzing up environments. In order to run this lesson 
please run `tests/demo_lesson_2r.py`. The particular reaction we will be simulating is performed in an aqueous state 
with ethoxyethane as the solvent.

From lesson 1 we know that our action space is a 6 element vector represented by:

|              | Temperature | Volume | 1-chlorohexane | 2-chlorohexane | 3-chlorohexane | Na  |
|--------------|-------------|--------|----------------|----------------|----------------|-----|
| Value range: | 0-1         | 0-1    | 0-1            | 0-1            | 0-1            | 0-1 |

The key to achieving a high reward in this simulation is to only add the reactants that are needed for the reaction to 
continue. This means that we will only add 3-chlorohexane and Na with our actions. This will allow us to maximize our 
reward as a large quantity of these reactants means the reaction with our target material will occur more often. We 
do this by running the following commands:

![code](../sample_figures/lesson_2r_image0.PNG)

Notice that we're only adding the reactants we need for the reaction to continue.

![reaction](../sample_figures/lesson_2r_image1.PNG)

Let's run our program and see what happens!

Notice that we get a total reward of 0.27 and if we write 'Y' in order to see the reaction vessel stats we should see 
that we produced a considerable amount of 4,5-diethyloctane. A visual representation of the reactants being used and 
total reward increasing can be seen in the subplot we produce!

![subplot](../sample_figures/lesson_2r_image3.PNG)

These are good results. Let's now run the same simulation but this time run the block of code that adds the reactants 
not needed.

![code](../sample_figures/lesson_2r_image4.PNG)

If we run this code we'll notice that our reward is significantly lower. It is now only 0.08 which is a drop-off from 
our previous set of actions. Once again, the reason this is happening is that other reactions are taking place instead 
of the reaction that produces our desired material. 

The next step for this reaction environment is to write an RL implementation that will allow the agent to solve this 
problem for you essentially maximizing our output of the desired material! 