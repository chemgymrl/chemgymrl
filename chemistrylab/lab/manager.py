import os
import sys

import numpy as np
import chemistrylab
import datetime as dt
from typing import NamedTuple, Tuple, Callable, Optional, List
from chemistrylab.benches.general_bench import *
from chemistrylab.benches.characterization_bench import CharacterizationBench
from chemistrylab.lab.shelf import Shelf
from chemistrylab.vessel import Vessel

class ManagerAction(NamedTuple):
    name: str
    args: dict
    cost: float
    bench: Optional[int]

class Manager():
    def __init__(self, benches: Tuple[GenBench], bench_names: Tuple[str], bench_agents: Tuple, targets: Tuple[str], actions: list = []):
        """
        Args:
            benches (Tuple[GenBench]): The benches available to this manager
            bench_names (Tuple[str]): A name for each bench.
            bench_agents (Tuple): A Tuple of (name, Policy) pairs.
            targets (Tuple[str]): A list of targets yo give to benches.
        """
        self.benches=benches
        self.bench_names=bench_names
        for b in self.benches:
            b.reset()
        self.hand = Shelf([],n_working=0)
        self.shelf = Shelf([],n_working=0)
        self.bench_agents = bench_agents
        self.targets=targets
    
    def swap_vessels(self, bench_idx, vessel_idx):
        """
        Swaps the vessel in hand with the a vessel in the shelf of one of the benches (or the lab manager's shelf).
        Note: This can add or remove vessels from a shelf depending on what is in the shelf / hand.

        Args:
            bench_idx (int): The index of the bench you want to modify the shelf of (set to -1 for the manager's shelf).
            vessel_idx (int): The shelf index you want to swap your vessel with.
        
        """
        if bench_idx<0:
            bench=self
        else:
            bench = self.benches[bench_idx]
        if len(self.hand)==1:
            if vessel_idx>=len(bench.shelf):
                bench.shelf.append(self.hand.pop())
            else:
                bench.shelf[vessel_idx], self.hand[0] = self.hand[0],bench.shelf[vessel_idx]
        elif len(self.hand)==0:
            if vessel_idx<len(bench.shelf):
                self.hand.append(bench.shelf.pop(vessel_idx))

    def dispose_vessel(self):
        if len(self.hand)>0:
            self.hand.pop()
    
    def create_vessel(self, label = ""):
        if len(self.hand)==0:
            self.hand.insert(0,Vessel(label))

    def insert_vessel(self, bench_idx, vessel_idx):
        """
        Inserts the vessel in the manager's hand into the benches shelf

        Args:
            bench_idx (int): The index of the bench you want to modify the shelf of (set to -1 for the manager's shelf).
            vessel_idx (int): The insertion index. (works the same as list.insert)
        
        """
        if len(self.hand)==0:return
        if bench_idx<0:
            bench = self
        else:
            bench = self.benches[bench_idx]

        bench.shelf.insert(vessel_idx, self.hand.pop())

    
    def set_target(self, bench_idx, target_idx):
        """
        Sets a benches target

        Args:
            bench_idx (int): The index of the bench you want to set the target in.
            target_idx (int): The target to set (corresponds to the managers list of targets).
        
        """
        self.benches[bench_idx].set_target(self.targets[target_idx])

    def restock_bench(self,bench_idx):
        """
        Restocks the shelf of a specified bench.

        NOTE: I may modify this later
        """
        bench =  self.benches[bench_idx]
        bench.shelf.restock(bench.get_target())

    def run_bench(self, bench_idx, policy_idx):
        """
        Wraps use_bench
        
        Args:
            bench_idx (GenBench): The index of the bench to run
            policy_idx (Policy): The index of the policy to use for the bench
        
        """
        self.use_bench(self.benches[bench_idx], self.policies[policy_idx])
    def use_bench(self, bench, policy):
        """
        Runs a bench with a given policy. (This will likely be modified in the future)

        Args:
            bench (GenBench): The bench to run
            policy (Policy): The Policy to use for the bench

        Returns:
            0 if the bench ran successfully and -1 otherwise.
        
        """
        #prep the bench
        #Check for an illegal vessel setup
        if not bench.validate_shelf():
            return -1

        o = bench.re_init()
        d = False
        while not d:
            o,r,d,*_ = bench.step(policy(o))
        
        for vessel in bench.shelf:
            print (vessel.get_material_dataframe())
            print (vessel.get_solute_dataframe())
            print(vessel._layers_settle_time)
            print(vessel._variance)
            print("-"*50)

        return 0
    
    def characterize(self, observation_list):
        """
        Provide an observation of the vessel in the manager's hand using a characterization bench

        Args:
            observation_list (Tuple[str]): A list of observations to make about the vessel
        
        """
        self.charbench = CharacterizationBench(observation_list,[],len(self.hand))
        return self.charbench(self.hand,"")

    def build_actions(self, actions: Tuple[ManagerAction]):
        self.current_bench = -1
        self.default_actions = [ManagerAction("set bench", dict(idx=i), 0 None) for i in range(len(self.benches))]
        self.bench_actions = [[] for bench in self.benches]
        for act in actions:
            if act.bench is None:
                self.default_actions.append(act)
            else:
                self.bench_actions[act.bench].append(act)

    def set_cur_bench(self, idx):
        self.current_bench = min(idx, len(self.benches)-1)

    def perform_action(self, action: ManagerAction):
        action_dict = type(self)._action_dict
        action_dict[action.name](self,**action.args)
        return action.cost

    _action_dict = {
        'swap vessels': swap_vessels,
        'dispose vessel': dispose_vessel,
        'create vessel': create_vessel,
        'insert vessel': insert_vessel,
        'set target': set_target,
        'restock bench': restock_bench,
        'run bench': run_bench,
        'characterize': characterize,
        'set bench': set_cur_bench,
    }



if __name__ == "__main__":
    import pygame
    from chemistrylab.lab.gui import *
    from chemistrylab.lab.manager_setup import parse_json
    import os
    demo_path = os.path.dirname(__file__)+"/manager_configs/wurtz.json"
    benches, bench_names, policies, targets, actions = parse_json(demo_path)

    # Set up visualization
    for i, env in enumerate(benches):
        for j in range(len(policies[i])):
            if type (policies[i][j][1]) is ManualPolicy:
                policies[i][j][1].set_env(env)
            else:
                policies[i][j] = (policies[i][j][0], VisualPolicy(env, policies[i][j][1]))

    manager = Manager(
        benches,
        bench_names,
        policies,
        targets
        )

    gui = ManagerGui(manager)
    while True:
        gui.render()
        if gui.input():
            break
        
