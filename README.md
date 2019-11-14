# chemistrygym
A  chemistry laboratory environment populated with a collection of chemistry experiment sub-environments, based on the OpenAI Gym environment framework for use with reinforcement learning applications.
## Vessel
### Important variables
  1. _material_dict: a dictionary holding all the material inside this vessel
  {'material.name': [material(), amount]}
  2. _solute_dict: dictionary that represents the solute
  {'solute.name': {solvent.name: amount}}
  3. _event_dict: a dictionary holds the event functions of a vessel
  {'temperature change': self._update_temperature,
                            'dissolve': self._dissolve,
                            }
  4. _event_queue: a queue of events to be performed by vessel
  5. _feedback_queue: a queue holding events feedback from materials and unfinished events
 
### Important functions
  1. push_event_to_queue(events, dt): called at the end of environment actions, events=None if no event generated
  2. _update_materials(dt): automatically called by push_event_to_queue(), excute events in _event_queue and _feedback_queue
  3. merge_event_queue(event_queue): merge the feedback_queue passed in and all the default event is appended to _feedback_queue by this function
 
##### How the 'event_queue' and 'feedback_queue' works
  
  1. Agent choose action from the action space of an environment.
  2. The enviroment do the calculation and update and etc.
  3. At the end of each action, the push_event_to_queue(events, dt) is called, if no event generated, events=None.
  4. _update_materials is automatically called and perform events in the events_queue.
  5. Each event has a corresponding event function, it first update properties of the vessel, then loop over the materials inside the vessel by calling the corresponding event funcitons of each material.
  6. The materials' event function will return feedback, which contains feedback and unfinished event 
  7. The returned feedback is added to the _feedback_queue
  8. The the merge_event_queue is called on _feedback_queue, which merge the events in the queue and add default event into it and empty the _feedback_queue and return a merged_queue
  9. Then the merged_queue will be performed and new feedback are collected and added to _feedback_queue, which will be performed after the next action. 