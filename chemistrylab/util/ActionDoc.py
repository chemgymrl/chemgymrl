import sys
from typing import List, Tuple, Optional
from tabulate import tabulate
from collections import namedtuple
from chemistrylab.benches.general_bench import Action, ContinuousParam
from chemistrylab.vessel import Vessel
from chemistrylab.benches.general_bench import GenBench


import numpy as np
floattypes = {np.float16, np.float32, np.float64, float}
Event = namedtuple('Event', ['name', 'parameter', 'other_vessel'])
Action = namedtuple('Action', ['vessels', 'parameters', 'event_name', 'affected_vessels', 'dt', 'terminal'])


param_names = {
    'pour by volume': ("volume",),
    'pour by percent': ("fraction",),
    'drain by pixel': ("n_pixel",),
    'mix': ("t",),
    'update_layer': (),
    'change heat': ("dQ",),
    'heat contact': ("Tf", "ht"),
}

def generate_table(shelf, actions) -> str:
    """Make a restructuredText table describing the actions list"""
    table_data = []
    for i, action in enumerate(actions, start=0):
        
        if action[1].terminal:
            vessels=event_params=other_vessel="N/A"
            event_performed = "End Experiment"
        else:
            event = action[0][0][1]
            event_performed = action[0][0][1].name
            
            discrete = True
            params = event.parameter
            if type(params) is ContinuousParam:
                discrete=False
                min_val, max_val, thresh, other = params
                params = (f"[{min_val},{max_val}]",*other)
            
            params = [round(a,3) if type(a) in floattypes else a for a in params]
            vessels = ', '.join([shelf[vessel[0]].label for vessel in action[0]])
            event_params = ', '.join([f"{name}: {a}" for name,a in zip(param_names[event.name],params)])
            other_vessel = event.other_vessel.label if event.other_vessel is not None else 'N/A'

        table_data.append([i, vessels, event_performed, event_params, other_vessel])
    
    action_col = "Action" if discrete else "Action Index"
    table_headers = [action_col, "Vessel", "Event Performed", "Event Parameters", "Other Vessel"]
    table = tabulate(table_data, headers=table_headers, tablefmt='grid')
    return table


def append_doc(bench):
    instance = bench()
    table = generate_table(instance.shelf,instance.actions)
    bench.__doc__+="Here is a breakdown of the action space:\n\n"+table+"\n\n"