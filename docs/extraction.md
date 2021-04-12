[chemgymrl.com](https://chemgymrl.com/)

## Extraction Bench

The extraction bench aims to isolate and extract certain materials from an inputted vessel container containing multiple materials. This is done by means of transferring materials between a number of vessels and utilizing specifically selected solutes to demarcate and separate materials from each other. The intended output of this bench is at least one beaker, each containing a desired material in quantities that exceed a minimum threshold.
 
A classic extraction experiment is the oil, alcohol, and water extraction experiment where oil is added to a beaker of water and alcohol settling above the water, but below the alcohol allowing the alcohol to be carefully poured into a secondary beaker. Similarly, in the extraction bench, solutes are added to act as the oil in the previous extraction experiment, separating collections of materials into layers that can be extracted and poured into auxiliary vessels. An agent performing this experiment will be able to use solutes and extract materials with the aim of extracting and isolating the specified desired material.
 
The agent operating on this bench will experiment using solutes in different scenarios and moving materials between beakers to learn which actions under which circumstances constitute positive and negative results. Positive results resemble actions that increase the purity of the desired material in a vessel or aim to increase the amount of the desired material in a vessel, while negative results include mixing the desired material with a host of unwanted materials. Like in the reaction bench, purity and amount are not the only variables used to facilitate the agent’s learning. Material costs and time constraints are additional factors that the agent must navigate when attempting to maximize the purity and yield of the desired material. Once the costs to continue outweigh the benefits or the required minimum purity threshold is exceeded, all vessels containing a sufficient amount of the desired material are outputted, stored, and made available for further use.

## Input 

The input to the extraction bench is initialized in the `extraction_bench_v1.py` file.

![extraction bench input](../tutorial_figures/extraction/extraction_bench_input.png)

Here we pass the extraction we want to perform, in the figure above, we would perform the wurtz extraction. The input 
vessel is also passed here. It contains the material and solute dictionary of a vessel that's typically outputted by the
reaction bench. We also pass the solute that the extraction will take place in, as well as the target material, and path 
of the output vessel.

## Output

The extraction bench plots data about the extraction being performed by the agent. Unlike reaction bench, in extraction 
there is only the human render mode.

![extraction output](../tutorial_figures/extraction/human_render_extraction.png)

The output plot shows the contents of each container and the level of separation between the materials. The graphs to 
the right shows the layers of materials forming in the container.

Like reaction bench the extraction bench also outputs a pickle file once the extraction process is completed. The
default name for this file is `extract_vessel_{i}` where i ranges from 0 to the total number of validated vessels.
