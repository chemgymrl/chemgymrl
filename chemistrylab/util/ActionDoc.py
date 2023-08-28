import sys
from typing import List, Tuple, Optional
from tabulate import tabulate
from collections import namedtuple
from chemistrylab.benches.general_bench import Action, ContinuousParam
from chemistrylab.vessel import Vessel
from chemistrylab.benches.general_bench import GenBench
from chemistrylab.lab.manager_setup import parse_json, ManagerAction
from chemistrylab.lab.manager import Manager

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


def args2string(**kwargs):
    return f"({', '.join(f'{k} = {v}' for k,v in kwargs.items())})"


def generate_manager_table(filename: str):
    benches, bench_names, policies, targets, actions = parse_json(filename)

    default_actions = [ManagerAction("set bench", dict(idx=i), 0,  None) for i in range(len(benches))]
    bench_actions = [[] for bench in benches]
    for act in actions:
        if act.bench is None:
            default_actions.append(act)
        else:
            bench_actions[act.bench].append(act)

    table_headers = ["Action Index","Required Bench", "Function Call"]

    table_data = []

    fdict = Manager._action_dict
    
    for i,act in enumerate(default_actions):
        table_data.append([i, "None", fdict[act.name].__name__+args2string(**act.args)])
    i=len(default_actions)

    for bench_idx, bench_name in enumerate(bench_names):
        for j,act in enumerate(bench_actions[bench_idx]):
            table_data.append([i+j, bench_name , fdict[act.name].__name__+args2string(**act.args)])

    table = tabulate(table_data, headers=table_headers, tablefmt='grid')
    return table

def append_doc(bench):
    instance = bench()
    table = generate_table(instance.shelf,instance.actions)
    bench.__doc__+="Here is a breakdown of the action space:\n\n"+table+"\n\n"