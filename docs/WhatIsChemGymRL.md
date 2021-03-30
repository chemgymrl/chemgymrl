[chemgymrl.com](https://chemgymrl.com/)

## What is ChemGymRL?

A chemistry laboratory environment populated with a collection of chemistry experiment sub-environments, based on the OpenAI Gym environment framework for use with reinforcement learning applications.

### Data Structure

volume: ml, amount: mole

### Bench Overview

The chemistrygym environment has three benches available: the reaction bench, the extraction bench, and the distillation bench. The reaction bench performs a series of reactions in a single vessel, the extraction bench moves materials between a series of vessels, and the distillation bench evaporates and isolates individual materials. Each bench utilizes vessels to contain materials and perform processes on the vessels. The reaction bench uses a single vessel while the extraction and distillation benches each use three vessels. Upon completing a bench, the final vessel(s), the vessel(s) containing the desired material or is of the most value, is(are) saved locally as a pickle file. Each bench can be initiated with an empty vessel(s) or be provided with a vessel(s) from previously completed benches. For the purposes of creating a realistic experiment, the workflow is intended to be the following.

reaction bench --> extraction bench --> distillation bench

Firstly, a series of reactions are performed in a reaction vessel and one material is designated as the "desired material". The reaction vessel is then saved and uploaded into the extraction bench. The extraction bench introduces at least one solute to allow for the individual movement of materials between three vessels. Any number of the final extraction vessels are saved and uploaded into the distillation bench. Note: more than one final extraction vessel may be of interest (ie. contains some amount of the desired material), but only one vessel can be loaded into the distillation bench at one time. The distillation bench applies heat energy to the uploaded vessel until a material begins to boil off into a secondary beaker and once the material has been completely boiled off that material is drained into a tertiary vessel. This process is repeated until the desired material has been completely isolated in its own vessel at which point the vessel is saved as it has 100% purity of the desired material.

### Reward Function

The reward function is crucial such that it is how the agents determine which actions are seen as positive and which actions are seen as negative. The current hierarchy of positive states has the amount of desired material present be of greater importance compared to the number of vessels containing the desired material. That being said, vessels containing of purity of less than 20%, with respect to the desired material, are considered unfit for continuation. For example, if extraction bench yields a vessel with 50% purity and another vessel with 10% purity, the vessel with 50% purity exceeds the purity requirement and is exempt from further actions in the extraction bench, while the vessel with 10% purity must still undergo extraction bench actions to increase the purity above the required purity threshold of 20%.