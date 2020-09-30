# Meeting Notes

## Date: August 4, 2020

### Attendees

- Chris Beeler
- Nouha Chatti (Secretary)
- Mark Crowley
- Sriram Ganapathi
- Mitchell Shahen 
- Isaac Tamblyn 

### Topic covered:

* Recap about last meeting, the reward function, test and demo from the different benches 
The extraction bench doesn't know what kind of molecule it's getting so It gets different reward if it's salt or butane 

* The expected behavior of the bench is independent of the content of the vessel as there's only information about the solvent. 

* The reward function is currently based on the amount and purity of salt 

* Isaac asked whether we can have general reward function of the polarity or the material property ? 

Can't be Polynomial , right now The end of the episode is the moles of the target material * the purity but It seems that the issue isn't with the reward, the agents needs to know whether it needs salt or butane. The agent doesn't know the task.

* Chris suggests adding the polarity of the target molecule to the observation space, could help the agent train on the scenarios it needs. In case it was given an impossible task it will do the same with salt but won't affect the policy so we could end up with an optimal policy 

* The order of steps and doing the procedure multiple times is used for optimization 
Minimize the cost based on how much the agent is penalized to use water for example for extracting salt.

* Lab manager is responsible for bearing the cost of an impossible task. He'll give the bench a task, the bench will run and return the result and be penalized accordingly and he'll return the cost in time.  

* Isaac suggests making the extraction bench more challenging and difficult. Adding an additional cost.

Chris's idea is to add something to the observation space to indicate the target and include how much is it in the vessel (lab manager would give an estimate)
Apply cost to the solvent (water/oil)
In the training procedure it will know how much material (salt) in the vessel 
So the agent will learn when to stop the extraction based on how much salt in there and cost of the material to perform the extraction.

Associating the lab manager to prior knowledge: (Minimum amount of salt/maximum)
So the idea is to send the min and max amount of each vessel to the bench as part of the task (or global min and max), Target molecule, information about the materials. 

The agent has also to learn how to use the polarities information.

* Mark suggests adding to the lab manager the Interpretation vector into the vessels that has as many elements as the materials in the vessel and updating this interpretation every time a vessel is passed through the lab manager. 

Store log of every experiment for the trajectory. The history that the agent can look at.
And when updating the perception : the lab manager reads through the history based on what it has seen and modifies the uncertainties.  

* Siriam:
Going through Curriculum learning paper 
The student could be the individual mdp 

* Chris: 
Discussed with Mitchell the benches 

Isaac asked if there's a clear analogy on other benches of what we discussed?

The answer is that for the reaction bench nothing to add other than the molecule and what is specific to the molecule 
For distillation bench, adding the boiling point should be enough 


* Next week : look at the lap manager loop, when the vessel comes back to the manager, calling the update interpretation methods (specific to the bench).

