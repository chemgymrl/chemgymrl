[chemgymrl.com](https://chemgymrl.com/)

## Distillation Bench

The distillation bench provides another set of experimentation aimed at isolating a requested desired material. Similar to the reaction and extraction benches, a vessel containing materials including the desired material is required as input into this bench. The distillation bench utilizes the differing boiling points of materials in the inputted vessel to separate materials between vessels. The intended output from the distillation bench is a vessel containing a sufficiently high purity and amount of the requested desired material.
 
A simple distillation experiment is boiling salt water thus evaporating the water into the air, or into a secondary vessel, leaving only the salt in the initial container. Similarly, the distillation bench obtains a vessel and gradually increases the vessel’s temperature incrementally boiling off materials one at a time. The materials, in their gaseous form, are deposited into an auxiliary vessel, which can be dumped into another auxiliary vessel for storage or removal.

An agent tasked to operate on this bench must control the heat energy added to the vessel as well as the movement of materials between the auxiliary vessels to isolate the desired material with as much purity as possible and spread about as few vessels as possible. Also required of the agent is monitoring the costs associated with adding heat energy and maintenance of unwanted materials. Positive and negative outcomes are associated with actions and operations that lead to the desired material being isolated and of high purity, and thoroughly mixed with unwanted materials and spread about several vessels, respectively.

## Input 

The input to the extraction bench is initialized in the `distillation_bench_v1.py` file.

![distillation bench input](../tutorial_figures/distillation/distillation_bench_input.PNG)

Here we pass the boiling vessel, which is typically the pickle file produced by the extraction bench. Like in the other 
engines we also pass the target material. Additionally, we pass in a dQ value which is the maximal change in heat 
energy and the path which the output vessel will be located in.

## Output

Like extraction, the distillation bench only has human render mode which renders a series of graphs illustrating the 
operations on the vessels. 

![distillation output](../tutorial_figures/distillation/human_render_distillation.png)

The top right graph shows the temperatures of the boiling vessel, beaker_0 and beaker_1. The other graphs plot the molar
amounts of each material in the vessel or beaker.

Like the other benches, distillation also saves the vessel once the distillation process is completed. The default name 
for the pickle file is 'distillation_vessel_{i}' where i ranges from 0 to the total number of validated vessels.
