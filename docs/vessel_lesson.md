[chemgymrl.com](https://chemgymrl.com/)

## Vessel: Lesson

### Overview

In this lesson, we will be going through a class that is vital to the operation of all of our benches, the vessel class.
The source code for this can be found here: `chemistrylab/chem_algorithms/vessel.py`. The vessel class as it is named is
meant to simulate the use of any given you might find in a chemistry lab, such as a beaker or an extraction vessel.
Here we will be going through the important concepts, functions and attributes that make up the vessel class so that you
can easily use it when designing your own reactions.

If you want a more detailed look into each function of the vessel I suggest you go to our documentation on the data structure. 

The Vessel class serves as any container you might find in a lab, a beaker, a dripper, etc. The vessel class simulates and allows for any action that you might want to perform within a lab, such as draining contents, storing gasses from a reaction, performing reactions, mix, pour, etc. This is performed using an event queue, which we will look at later in this lesson. First an overview of some of the important variables that make up the vessel class:

Important Variables |Structure | Description
---|---|---
_material_dict|{'material.name': [material(), amount, unit], ...}|a dictionary holding all the material inside this vessel
_solute_dict|{'solute.name': {solvent.name: [solvent class, amount, unit], ...}, ...}|dictionary that represents the solution
_event_dict|{'function name': function}|a dictionary holds the event functions of a vessel
_event_queue|[['event', parameters], ['event', parameters] ... ]|a queue of events to be performed by vessel
_feedback_queue|[['event', parameters], ['event', parameters] ... ]|a queue holding collected feedback from materials and unfinished events


#### An example of _material_dict and _solute_dict
(vessel with Na and Cl dissolved in H2O and C6H14)


```python
import sys
sys.path.append('../')
sys.path.append('../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np
from gym import envs
from chemistrylab.chem_algorithms import material, vessel
H2O = material.H2O()
C6H14 = material.C6H14()
Na = material.Na()
Cl = material.Cl()

Na.set_charge(1.0)
Na.set_solute_flag(True)
Cl.set_charge(-1.0)
Cl.set_solute_flag(True)

material_dict = {'H2O': [H2O, 100, 'mol'], 'C6H14': [C6H14, 100], 'Na': [Na, 1.0], 'Cl': [Cl, 1.0]}
solute_dict = {'Na': {'H2O': [H2O, 0.5, 'mol'], 'C6H14': [C6H14, 0.5, 'mol']}, 'Cl': {'H2O': [H2O, 0.5, 'mol'], 'C6H14': [C6H14, 0.5, 'mol']}}
```

To briefly describe above, the material_dict describes the materials contained within a vessel and the quantity of that material. The material dict is a dictionary with the key as the name of the material and whose item is a list consisting of the material class, the quantity of the material and optionally a unit, if a unit is not provided we assume the use of moles. As for the solute dict, the solute dict represents how much of the material is dissolved in each material. So the solute dict is first indexed by the solute, which contains another dictionary of its solvents, then when we index the solvent we get what percentage of the solution is dissolved in each material. Above we can see that sodium and chlorine are both dissolved in 50% water and 50% oil.


Next we will look at some of the important functions that we will need to use with the vessel class:

Important functions | Description
---|---
push_event_to_queue()|used to pass event into the vessel
_update_materials()|automatically called by push_event_to_queue(), execute events in _event_queue and _feedback_queue
_merge_event_queue()|merge the feedback_queue passed in, and all the default events (switches) are appended to _feedback_queue by this function


From the list above, the most important function is push_event_to_queue(). The rest of the functions are handeled in the backend by the events queue. As stated before, the vessel class works off of an event queue where we push a set of events to the event queue and the event queue then performs those events to the vessel, along with a specified dt parameter to specify how much time to wait after performing the events. Below we have a list of the current event functions that can be performed by the vessel and below that we have a demonstration of how to push events to the vessel.


### Event Functions
Function Name|Description
---|---
'pour by volume'|Pour from self vessel to target vessel by certain volume
'drain by pixel|Drain from self vessel to target vessel by certain pixel
'update material dict'|Use input to update self vessel's material_dict
'update solute dict'|Use input to update self vessel's solute_dict
'mix'|Shake vessel of let vessel settle
'update_layer'|Update self vessel's layer representation




```python
vessel_1 = vessel.Vessel(label='vessel_1')
vessel_2 = vessel.Vessel(label='vessel_2')

event_1 = ['update material dict', material_dict]
event_2 = ['update solute dict', solute_dict]
event_3 = ['drain by pixel', vessel_2, 100]

# Here we are adding the materials and solutions specified above into our first vessel, and then we pour 100ml
# of this solution from vessel 1 into vessel 2
vessel_1.push_event_to_queue(events=None, feedback=[event_1, event_2, event_3], dt=1)
```

To visualize what we are doing above, we have several graphs:

Here we have a graph of the seperation between the oil and the water when we initially add them to our vessel

![added oil and water](../tutorial_figures/vessel/vessel_1.png)

Here we have the solution after we drain some of it (notice that the top of the vessel is now air):

![drained oil and water](../tutorial_figures/vessel/vessel_2.png)
### The Workflow
  
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

```python

```
