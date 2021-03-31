[chemgymrl.com](https://chemgymrl.com/)

## Reaction Bench

The reaction bench is intended to take a list of materials or imputed vessels and perform a series of reactions on the materials in that vessel, step by step. The output should be a vessel containing products that will be available for extraction and distillation. We take a vessel that is frozen in time, and that has not been allowed to react or evolve in any way, and once it is placed in a bench we allow time to flow, and as time goes by, reactions occur within the vessel. Once the agent deems that the reactions are to be stopped, time is frozen again and the vessel is put on a shelf, ready to be used by the other benches. 

In terms of the implementation strategy, the way this bench works is an agent selects a family of reactions from reactions stored in the environment to occur on the materials present in the vessel. Eventually, the goal is for the reaction bench to look at the materials in the vessel and performs every possible reaction that could occur instead of having the agent perform a select few reactions. The reactions happen for a set amount of time determined by the amount of material and the type of material in the vessel, which the user can use in their favor to control the amount of product created by the bench. 

The rewards system allows for the agent to determine what are positive and negative outcomes and can be used to optimize actions in the reaction process and help the user obtain their desired product. Every reaction is done in order to obtain a desired product, as would occur in real life. The agent is then told to maximize the output of said desired product, and the reward system is what gets the agent to perform optimal reactions until the output is satisfactory.
