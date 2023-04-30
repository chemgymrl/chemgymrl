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
from chemistrylab.chem_algorithms import vessel
from chemistrylab.reactions.reaction import Reaction
from chemistrylab.characterization_bench.characterization_bench import CharacterizationBench

class Action(NamedTuple):
    vessels: Tuple[int]
    parameters: Tuple[tuple]
    event_name: str
    affected_vessels: Optional[Tuple[int]]
    dt: float
    terminal: bool
    
class Event(NamedTuple):
    name: str
    parameter: tuple
    other_vessel: Optional[object]
        
class ContinuousParam(NamedTuple):
    min_val: float
    max_val: float
    thresh: float
    other: object

def default_reward(vessels,targ):
    sum_=0
    for vessel in vessels:
        mats=vessel.material_dict
        all_mat = sum(mats[a].mol for a in mats)
        if all_mat>1e-12:
            sum_+=(mats[targ].mol if targ in mats else 0)**2/all_mat
    return sum_
    
class GenBench(gym.Env):
    """A class representing an bench setup for conducting experiments.

    This class represents a bench setup. The setup consists of a
    set of vessels, an action list that describes the actions to be performed on
    the vessels, and a Reaction class simulating the chemical reactions taking place.

    Args:
    - vessel_generators (tuple): A tuple of generators, each of which yield a Vessel object.
    - action_list (list): A list of Action objects containing information describing what each action is.
    - reaction_info (ReactInfo): A ReactInfo object that provides information about the chemical reactions taking place.
    - n_visible (Optional int): The number of vessels (first n) which will be visible in the observation space
    - react_list (Tuple[int]): Which vessels reactions should take place in (leave empty for no reacting)
    - targets (Tuple[str]): The target materials for this bench
    - discrete (bool): set to True for a discrete action space and False for a continuous one
    Keyword Args:
    - kwargs: To be determined based on need.

    """
    def __init__(
        self,
        vessel_generators: Callable,
        action_list: Tuple[Action],
        reaction_info,#work in progress
        observation_list: Tuple[str],
        n_visible: Optional[int] = None,
        reward_function: Callable = default_reward,
        react_list: Optional[Tuple[int]] = None,
        targets: Optional[Tuple[str]] = None,
        discrete=True,
        max_steps=50,
        **kwargs
    ):
        
        self.n_visible = len(vessel_generators) if n_visible is None else n_visible
        #set up reactions
        self.react_list = [] if react_list is None else react_list
        self.reaction=Reaction(reaction_info)
        
        # store bench information
        self.vessel_generators=vessel_generators
        self.action_list=action_list
        self.reaction_info=reaction_info
        self.reward_function=reward_function
        self.max_steps=max_steps
        #Making sure targets are populated
        self.targets = targets
        if self.targets is None:
            self.targets = self.reaction_info.PRODUCTS
        
        #get counts
        self.num_targets=len(self.targets)
        self.n_actions=sum(1 for a in action_list for p in a.parameters)

        
        #Set up observation and action space
        self.characterization_bench =  CharacterizationBench(observation_list,self.targets,self.n_visible)
        self.observation_space = gym.spaces.Box(0,1,self.characterization_bench.observation_shape, dtype=np.float32)
        
        self.discrete=discrete
        if self.discrete:
            self.action_space = gym.spaces.Discrete(self.n_actions)
        else:
            self.action_space = gym.spaces.Box(0, 1, (self.n_actions,), dtype=np.float32)
        
        self.reset()
        
    def get_vessels(self):
        return self.vessels
    def update_vessels(self,new_vessels):
        self.vessels=new_vessels
        
    def build_event(self,action,param):
        """
        Builds a list of (index, Event) tuples where index is the index of the vessel the Event will occur in.

        Args:
        - action (Action): An action containing information about the vessels and parameters
                involved in the event.
        - param (tuple): A tuple containing parameters for the event.

        Returns:
        - events (List[Tuple]): A list of (index, Event) tuples, where index is the index of the vessel the Event will occur in.
        """
        #watch out for null case
        if action.affected_vessels is None:
            other_vessels = [None]*len(action.vessels)  
        else:
            other_vessels = [self.vessels[i] for i in action.affected_vessels]
        
        #I may change this in the future to remove support for an action to span multiple vessel pairs
        return [(v,Event(action.event_name,param,other_vessels[i])) for i,v in enumerate(action.vessels)]
    
    def _perform_continuous_action(self,action):
        """
        Action should be a 1D array.
        
        Here, event parameters follow the ContinuousParam format. 
        In this implementation all actions are performed with dt=0, then afterwards
        all visible vessels are given an empty event with dt=0.01. 
        (May create a variable to replace 0.01)
        """
        for i,(events,_action) in enumerate(self.actions):
            val=action[i]
            #handling any end of episode actions
            if _action.terminal and (val>events[0][1].parameter.thresh):
                return True
            
            #update the involved vessels
            for v,event in events:
                # move the action in [0,1] to [min_val,max_val]
                min_val, max_val, thresh, other = event.parameter
                activ=val*(val>thresh)
                rescaled = activ*(max_val-min_val)+min_val
                #create a new event with this
                param=(rescaled,other)
                derived_event = Event(event.name,param,event.other_vessel)
                #perform the new event
                self.vessels[v].push_event_to_queue(events=[derived_event], dt=0.0)
        #all vessels which appear in the observation space are updated
        for v in range(self.n_visible):
            self.vessels[v].push_event_to_queue(dt=0.01)
        return False
    def _perform_discrete_action(self,action):
        """
        Action should be an integer.
        Here, the chosen action is perfomed on all relevant vessels, then all
        other visible vessels are fed an empty event with the same dt.
        """
        events,_action = self.actions[action]
        updated=set()
        #update the involved vessels
        for v,event in events:
            updated.add(v)
            #going to hard-code dt for now
            self.vessels[v].push_event_to_queue(events=[event], dt=_action.dt)
        #all uninvolved vessels which appear in the observation space
        for v in range(self.n_visible):
            if not v in updated:
                self.vessels[v].push_event_to_queue(dt=_action.dt)
        
        return _action.terminal
    
    def step(self,action):
        """
        Here, actions are performed by '_perform_discrete_action' or '_perform_continuous_action'. Afterwards, all vessels in
        react_list have their concentrations updated by a reaction (specified in __init__). Finally, a CharacterizationBench
        is used to generate an observation, and a reward function provides a reward (if a terminal state was reached).
        
        Args:
        - action (int or 1D array): The action to be performed
        """
        if self.discrete:
            done = self._perform_discrete_action(action)
        else:
            done = self._perform_continuous_action(action)
            
        #perform any reactions
        for v in self.react_list:
            self.reaction.update_concentrations(self.vessels[v])
            
        #Increment the step counter and check if you are done
        self.steps+=1
        done=done or self.steps>=self.max_steps
        
        #Handle reward
        reward=0
        if done:
            reward = self.reward_function(self.vessels[:self.n_visible],self.target_material)-self.initial_reward
        
        #gather observations
        state=self.characterization_bench(self.vessels[:self.n_visible],self.target_material)
        
        return state, reward, done, {}
    
    def _reset(self,target):
        self.steps=0
        self.vessels = [v(target) for v in self.vessel_generators]
        self.target_material=target
        #Rebuild the list of actions with the new vessels
        self.actions = [(self.build_event(a,p),a)  for a in self.action_list for p in a.parameters]
        #Gather the initial reward using a provided reward function
        self.initial_reward = self.reward_function(self.vessels[:self.n_visible],self.target_material)
        
        return self.characterization_bench(self.vessels[:self.n_visible],self.target_material)
    
    def reset(self):
        target=np.random.choice(self.targets)
        return self._reset(target)
        
        