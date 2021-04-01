[chemgymrl.com](https://chemgymrl.com/)

## Reaction Bench

The reaction bench is intended to take a list of materials or inputted vessels and perform a series of reactions on the materials in that vessel, step by step. The output will be a vessel containing products that will be available for extraction and distillation. 

We take a vessel that is frozen in time, and that has not been allowed to react or evolve in any way, and once it is placed in a bench we allow time to flow, and as time goes by, reactions occur within the vessel. Once the agent deems that the reactions are to be stopped, time is frozen again and the vessel is put on a shelf, ready to be used by the other benches. 

In terms of the implementation strategy, the way this bench works is an agent selects a family of reactions from reactions stored in the environment, and in order for these reactions to occur, the vessel must have the required materials for the chosen reactions. The reactions occur over a number of steps, each occuring in a set amount of time. The number of steps used is determined by the agent working at the reaction bench. When an agent recieves a sufficiently large reward from performing numerous positive tasks, then the agent may choose to end the experiment in the reaction bench.. 

The rewards system reinforces good and bad outcomes by looking at the vessel after each step and can be used to optimize actions in the reaction process and help the user obtain their desired product. Every reaction is done in order to obtain a desired product, as would occur in real life. The agent is then told to maximize the output of said desired product, and the reward system is what gets the agent to determine whether or not an episode is to be concluded or not, based on whether or not the output is satisfactory.
