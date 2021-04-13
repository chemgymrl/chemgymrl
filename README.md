# ChemGymRL

[![Build Status](https://travis-ci.com/chemgymrl/chemgymrl.svg?branch=main)](https://travis-ci.com/chemgymrl/chemgymrl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chemgymrl.readthedocs.io/en/latest/)

## Overview

ChemGymRL is a collection of reinforcement learning environments in the open AI standard which are designed for experimentation and exploration with reinforcement learning in the context of chemical discovery. These experiments are virtual variants of experiments and processes that would otherwise be performed in real-world chemistry labs and in industry.

ChemGymRL simulates real-world chemistry experiments by allocating agents to benches to perform said experiments as dictated by a lab manager who has operational control over all agents in the environment. The environment supports the training of Reinforcement Learning agents by associating positive and negative rewards based on the procedure and outcomes of actions taken by the agents.

## Data Structure

volume: ml, amount: mole

### Bench Overview

The ChemGymRL environment consists of four benches and a lab manager tasked with organizing the experimentation occurring at each bench. The four benches available to agents under the direction of the lab manager include: the reaction bench, the extraction bench, the distillation bench, and the characterization bench.

The ChemGymRL environment aims to simulate laboratory processes performed by agents trained by means of reinforcement learning algorithms. The goal of the ChemGymRL environment, similar to many laboratory processes and experiments, is to produce, isolate, and extract a valuable material. The lab manager is instructed by the user to acquire such a material and the lab manager dispatches agents to the various benches to do so.

Each bench contributes to this goal by providing experimentation methods that, when used in tandem, can produce the desired output. The reaction bench performs a series of reactions yielding various products. The extraction bench allows for the extraction of certain materials that would otherwise be mixed together with unwanted byproducts. The distillation bench heats up materials allowing certain materials to be extracted in their gaseous forms. The characterization bench allows the agents and lab managers to inspect containers of materials and determine their contents.

### Reward Function

The reward function is crucial such that it is how the agents determine which actions are seen as positive and which actions are seen as negative. The current hierarchy of positive states has the amount of desired material present be of greater importance compared to the number of vessels containing the desired material. That being said, vessels containing of purity of less than 20%, with respect to the desired material, are considered unfit for continuation. For example, if extraction bench yields a vessel with 50% purity and another vessel with 10% purity, the vessel with 50% purity exceeds the purity requirement and is exempt from further actions in the extraction bench, while the vessel with 10% purity must still undergo extraction bench actions to increase the purity above the required purity threshold of 20%.

### Vessel

Important Variables |Structure | Description
---|---|---
_material_dict|{'material.name': [material(), amount], ...}|a dictionary holding all the material inside this vessel
_solute_dict|{'solute.name': {solvent.name: amount, ...}, ...}|dictionary that represents the solution
_event_dict|{'function name': function}|a dictionary holds the event functions of a vessel
_event_queue|[['event', parameters], ['event', parameters] ... ]|a queue of events to be performed by vessel
_feedback_queue|[['event', parameters], ['event', parameters] ... ]|a queue holding collected feedback from materials and unfinished events

#### The Workflow
  
  1. Agent choose action from the action space of an environment.
  2. The environment does the calculation and update and generate events.
  3. At the end of each action, if the action affect a vessel, use push_event_to_queue() to push the event into the vessel, if no event generated, call the function with events=None.
  4. With push_event_to_queue() called, events are pushed into the vessel.
  5. _update_materials is automatically called and perform events in the events_queue.
  6. Each event has a corresponding event function, it first update properties of the vessel, then loop over the materials inside the vessel by calling the corresponding event functions of each material.
  7. The materials' event function will return feedback by calling the push_event_to_queue(), which contains feedback and unfinished event 
  8. The returned feedback is added to the _feedback_queue
  9. The the _merge_event_queue() is called on _feedback_queue, which merge the events in the feedback_queue to generate a merged_queue and add default event into it, then empty the _feedback_queue
  10. Then the merged_queue will be executed and new feedback are collected and added to _feedback_queue, which will be executed with the next action. 

#### Event Functions

Function Name|Description
---|---
'pour by volume'|Pour from self vessel to target vessel by certain volume
'drain by pixel|Drain from self vessel to target vessel by certain pixel
'fully mix'|Shake self vessel to fully mix
'update material dict'|Use input to update self vessel's material_dict
'update solute dict'|Use input to update self vessel's solute_dict
'mix'|Shake vessel of let vessel settle
'update_layer'|Update self vessel's layer representation

### Material

Properties|Description
---|---
name|Name of the material
density|Density of the material, g/mL
polarity|Polarity of the material
temperature|Temperature of the material, K
pressure|Pressure of the material, kPa
phase|Phase of the material (g, l, s)
charge|Charge of the material
molar_mass|Molar mass of the material, g/mol
color|A unique float number between 0 to 1 to represent the color of the the material, used in layer representation
solute|A flag showing if it's the solute in vessel (True/False)
solvent|A flag showing if it's the solvent in vessel (True/False)
index|A integer number, unique for each material, used to fix the location of the material in state (matrix)
 

### Util Functions

Function|Description
---|---
convert_material_dict_to_volume|Convert the input material_dict to a dictionary of {material's name: volume}
convert_volume_to_mole|Handy function to calculate mole
organize_material_dict|Organize the input material_dict, called by update_material_dict()
organize_solute_dict|Organize the input solute_dict, called by update_solute_dict()
check_overflow|Check if there's overflow, and calculate what remains in the vessel (always call it before push new material_dict and solute_dict to target vessel)
generate_state|generate the state representation from a list of vessels

### State Structure

Each vessel has three matrices to represent the state:

[material_dict_matrix, solute_dict_matrix, layer_vector]

**material_dict_matrix (number of available material * 10):**

-|Density|Polarity|Temperature|Pressure|Solid Flag|Liquid Flag|Gas Flag|Charge|Molar Mass|Amount
---|---|---|---|---|---|---|---|---|---|---
Air|
H2O|
H|
H2|
.|
.|
.|

**solute_dict_matrix (number of available material * number of available material):**

Each cell means how much of that solute (row) dissolved in that solvent (column)

-|Air|H2O|H|H2|.|.|.
---|---|---|---|---|---|---|----
Air|
H2O|
H|
H2|
.|
.|
.|

**layer_vector (1 * number of pixel):**

acquired form vessel.get_layers()


