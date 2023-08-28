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
import gymnasium as gym
from chemistrylab.util.reward import RewardGenerator
from chemistrylab.lab.manager_setup import parse_json

CONFIG_PATH = os.path.dirname(__file__) + "/manager_configs"

class ManagerAction(NamedTuple):
    name: str
    args: dict
    cost: float
    bench: Optional[int]

class Manager(gym.Env):
    def __init__(self, config_filename: str):
        if config_filename is None:
            print("call to setup() is Required")
        else:
            benches, bench_names, policies, targets, actions = parse_json(config_filename)
            self.setup(
                benches,
                bench_names,
                policies,
                targets,
                actions
            )
    
    def setup(self, benches: Tuple[GenBench], bench_names: Tuple[str], bench_agents: Tuple, targets: Tuple[str], actions: list = []):
        """
        Args:
            benches (Tuple[GenBench]): The benches available to this manager
            bench_names (Tuple[str]): A name for each bench.
            bench_agents (Tuple): A Tuple of (name, Policy) pairs.
            targets (Tuple[str]): A list of targets yo give to benches.
            actions (Tuple[ManagerAction]): A list of actions the bench can perform
        """
        self.benches=benches
        self.bench_names=bench_names
        for b in self.benches:
            b.reset()
            #reset to empty target
            b.reset_with_target("")
        self.hand = Shelf([],n_working=0)
        self.shelf = Shelf([],n_working=0)
        self.bench_agents = bench_agents
        self.targets=targets
        self.build_actions(actions)
        self.current_bench = None

        self.action_space = gym.spaces.Discrete(self.n_actions)

        observation_list=[]
        for act in actions:
            if act.name == "characterize":
                observation_list+= act.args["observation_list"]

        self.char_bench = CharacterizationBench(observation_list,self.targets,1)
        self.hand_encoding = np.zeros(self.char_bench.observation_shape)


        #TODO: change this
        self.observation_space = gym.spaces.Box(0,1,
            (len(self.targets)+len(self.benches)+11+self.char_bench.observation_shape[0],), 
            dtype=np.float32
        )
        self.rew_gen = RewardGenerator(use_purity=True,exclude_solvents=False,include_dissolved=True)



        self.steps=0
        self.max_step = 100
        self.target_material = ""
    



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

        self.hand_encoding = np.zeros(self.char_bench.observation_shape)

    def dispose_vessel(self):
        """
        Removes and deletes the vessel in the manager's hand
        """
        if len(self.hand)>0:
            self.hand.pop()
        self.hand_encoding = np.zeros(self.char_bench.observation_shape)
    
    def create_vessel(self, label = ""):
        """
        Creates an empty vessel and places it in the manager's hand (as long as the hand is empty)

        Args:
            label (str): A name for the vessel
        """
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

        self.hand_encoding = np.zeros(self.char_bench.observation_shape)
    
    def set_bench_target(self, bench_idx, target_idx):
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
        self.use_bench(self.benches[bench_idx], self.bench_agents[bench_idx][policy_idx][1])
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
        
        for vessel in []:#bench.shelf:
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
        #self.charbench = CharacterizationBench(observation_list,[],len(self.hand))
        #update = self.charbench(self.hand,"")
        if len(self.hand)==0:return
        update, mask = self.char_bench.get_observation_subset(self.hand, observation_list, "")

        self.hand_encoding = self.hand_encoding*(1-mask)+update

        return update[mask!=0]


    def encode_bench(self):
        """
        Returns:
            np.ndarray: A vector containing a one-hot encoding of which bench is selected, as well as some information about the contents of the benches shelf.
        """
        encoding = np.zeros(len(self.benches)+10)
        if self.current_bench is None:
            return encoding
        encoding[self.current_bench] = 1
        idx=3
        for vessel in self.benches[self.current_bench].shelf[:10]:
            encoding[idx]=vessel.filled_volume()/vessel.volume
            idx+=1
        return encoding

    def get_observation(self):
        """
        Creates an observation of the manager's state
        
        TODO: Give info about the manager's shelf

        Returns:
            np.ndarray: The observation
        
        """
        return np.concatenate([self.target_encoding, self.encode_bench(),[len(self.hand)],self.hand_encoding])

    def build_actions(self, actions: Tuple[ManagerAction]):
        """
        Construct the list of actions (different for each bench)
        
        Args:
            actions (Tuple[ManagerAction]): A list of actions the bench can perform
        """
        self.current_bench = -1
        self.default_actions = [ManagerAction("set bench", dict(idx=i), 0,  None) for i in range(len(self.benches))]
        self.bench_actions = [[] for bench in self.benches]
        for act in actions:
            if act.bench is None:
                self.default_actions.append(act)
            else:
                self.bench_actions[act.bench].append(act)

        self.n_actions = len(self.default_actions)+max(len(a) for a in self.bench_actions)

    def set_cur_bench(self, idx):
        """
        Sets the bench the manager is currently focused on.
        
        Args:
            idx (int): The bench index
        """
        self.current_bench = min(idx, len(self.benches)-1)

    def end_experiment(self):
        """
        After calling this, the next call to step() will result in a terminal state.
        """
        self.steps = self.max_step

    def _perform_action(self, action: ManagerAction):
        action_dict = type(self)._action_dict
        action_dict[action.name](self,**action.args)
        return action.cost

    def step(self, action: int):

        cost = 0
        self.steps+=1
        if action < len(self.default_actions):
            self._perform_action(self.default_actions[action])
            cost = self.default_actions[action].cost

        elif self.current_bench is not None:
            action -= len(self.default_actions)
            if action < len(self.bench_actions[self.current_bench]):
                self._perform_action(self.bench_actions[self.current_bench][action])
                cost = self.bench_actions[self.current_bench][action].cost
        
        done = self.steps>=self.max_step
        rew = 0 if not done else self.rew_gen(self.shelf,self.target_material)
        return self.get_observation(), rew-cost, done, False, {}

    def reset_with_target(self,target: str):
        """
        Resets the bench with a specific target material.

        Args:
            target (str): The target material
        """
        for b in self.benches:
            b.reset()
            #reset to empty target
            b.reset_with_target("")
        self.hand = Shelf([],n_working=0)
        self.shelf = Shelf([],n_working=0)
        self.steps = 0
        self.set_target(target)

        return self.get_observation()
    
    def set_target(self,target):
        """
        Sets the manager's target material

        Args:
            target (str): The smiles code of the target material
        
        """
        self.target_material=target
        self.char_bench.target = target
        self.target_encoding = self.char_bench.encode_target(None)
    def get_target(self):
        """
        Returns:
            str: The smiles code of the manager's target
        
        """
        return self.target_material
    
    def reset(self, *args, seed=None, options=None):

        np.random.seed(seed)
        self.action_space.seed(seed)
        target=np.random.choice(self.targets)
        return self.reset_with_target(target),{}

    _action_dict = {
        'swap vessels': swap_vessels,
        'dispose vessel': dispose_vessel,
        'create vessel': create_vessel,
        'insert vessel': insert_vessel,
        'set bench target': set_bench_target,
        'restock bench': restock_bench,
        'run bench': run_bench,
        'characterize': characterize,
        'set bench': set_cur_bench,
        'end experiment': end_experiment,
    }



if __name__ == "__main__":
    import pygame
    from chemistrylab.lab.gui import *
    
    import os
    demo_path = os.path.dirname(__file__)+"/manager_configs/demo.json"
    benches, bench_names, policies, targets, actions = parse_json(demo_path)

    # Set up visualization
    for i, env in enumerate(benches):
        for j in range(len(policies[i])):
            if type (policies[i][j][1]) is ManualPolicy:
                policies[i][j][1].set_env(env)
            else:
                policies[i][j] = (policies[i][j][0], VisualPolicy(env, policies[i][j][1]))

    manager = Manager(None)
    manager.setup(
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
        
