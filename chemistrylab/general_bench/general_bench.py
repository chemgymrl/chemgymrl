"""
The aim of this bench is to replace extraction and distillation benches with one where
you can set the actions using a dictionary or json file.

"""

#useful imports
from typing import NamedTuple, Tuple, Callable, Optional
import gym
import numpy as np

#Imports which need to go soon
import sys
sys.path.append("../../") # to access chemistrylab
from chemistrylab.chem_algorithms import util, vessel
from chemistrylab.reactions.reaction import Reaction


class Bench(gym.Env):
    def get_vessels(self):
        raise NotImplementedError("Bench is abstract class")
    
    def update_vessels(self, new_vessels):
        raise NotImplementedError("Bench is abstract class")
    
    def step(self,act):
        raise NotImplementedError("Bench is abstract class")
    
    def reset(self):
        raise NotImplementedError("Bench is abstract class")
        
        
class Action(NamedTuple):
    vessels: Tuple[int]
    parameters: Tuple[tuple]
    event_name: str
    affected_vessels: Optional[Tuple[int]]
    terminal: bool

class Event(NamedTuple):
    name: str
    parameter: tuple
    other_vessel: Optional[object]   

def default_reward(vessels,targ):
    sum_=0
    for vessel in vessels:
        mats=vessel._material_dict
        sum_+=mats.get(targ,(0,0))[1]**2/sum(mats[a][1] for a in mats)
    
class GenBench(Bench):
    """A class representing an bench setup for conducting experiments.

    This class represents a bench setup. The setup consists of a
    set of vessels, an action list that describes the actions to be performed on
    the vessels, and a Reaction class simulating the chemical reactions taking place.

    Args:
    - vessel_generators (tuple): A tuple of generators, each of which yield a Vessel object.
    - action_list (list): A list of Action objects containing information describing what each action is.
    - reaction_info (ReactInfo): A ReactInfo object that provides information about the chemical reactions taking
            place.
    - n_visible (Optional int): The number of vessels (first n) which will be visible in the observation space

    Keyword Args:
        kwargs: To be determined based on need.

    """
    def __init__(
        self,
        vessel_generators: Callable,
        action_list: Tuple[Action],
        reaction_info,#work in progress
        n_visible: Optional[int] = None,
        reward_function: Callable = default_reward,
        react_list=None,
        targets=None,
        **kwargs
    ):
        
        self.n_visible = len(vessel_generators) if n_visible is None else n_visible
        
        self.react_list = [] if react_list is None else react_list
        self.reaction=Reaction(reaction_info)
        
        self.vessel_generators=vessel_generators
        self.action_list=action_list
        self.reaction_info=reaction_info
        self.reward_function=reward_function
        
        self.n_actions=sum(1 for a in action_list for p in a.parameters)
        
        self.targets = targets
        if self.targets is None:
            self.targets = self.reaction_info.PRODUCTS
        self.num_targets=len(self.targets)
        
        #This block will have to change later to accomodate other observation spaces (right now I used extract bench)
        self.n_pixels=100
        obs_high = np.ones((self.n_visible, self.n_pixels + self.num_targets), dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_high*0, obs_high, dtype=np.float32)
        
        #only supporting discrete action spaces right now
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        self.reset()
        
    def get_vessels(self):
        return self.vessels
    def update_vessels(self,new_vessels):
        self.vessels=new_vessels
        
    def get_state(self):
        state = np.zeros((self.n_visible, self.n_pixels + self.num_targets), dtype=np.float32)
        #layer observation (used for extract and distill benches)
        state[:, :self.n_pixels] = util.generate_layers_obs(
            vessel_list=self.vessels[:self.n_visible],
            max_n_vessel=self.n_visible,
            n_vessel_pixels=self.n_pixels
        )
        #target is like this in all benches
        targ_ind = self.targets.index(self.target_material)
        state[:, self.n_pixels + targ_ind] = 1
        
        return state

    def build_event(self,action,param):
        """
        Returns a tuple of (index,Event) where index is the index of the vessel the Event will occur in
        """
        #watch out for null case
        if action.affected_vessels is None:
            other_vessels = [None]*len(action.vessels)  
        else:
            other_vessels = [self.vessels[i] for i in action.affected_vessels]

        return [(v,Event(action.event_name,param,other_vessels[i])) for i,v in enumerate(action.vessels)]
    
    def step(self,action):
        act,_action = self.actions[action]
        updated=set()
        #update the involved vessels
        for v,event in act:
            updated.add(v)
            #going to hard-code dt for now
            self.vessels[v].push_event_to_queue(events=[event], dt=0.01)
        #all uninvolved vessels which appear in the observation space
        for v in range(self.n_visible):
            if not v in updated:
                self.vessels[v].push_event_to_queue(dt=0.01)
                
        for v in self.react_list:
            self.reaction.update_concentrations(self.vessels[v])
        #Handle reward
        reward=0
        if _action.terminal:
            reward = self.reward_function(self.vessels[:self.n_visible],self.target_material)-self.initial_reward
        
        return self.get_state(), reward, _action.terminal, {}
    
    def _reset(self,target):
        self.vessels = [v(target) for v in self.vessel_generators]
        self.target_material=target
        
        #might rethink this line
        self.actions = [(self.build_event(a,p),a)  for a in self.action_list for p in a.parameters]
        
        
        #Reward (will probably change the reward system later)
        self.initial_reward = self.reward_function(self.vessels[:self.n_visible],self.target_material)
        
        return self.get_state()
    
    def reset(self):
        target=np.random.choice(self.targets)
        return self._reset(target)
        
        