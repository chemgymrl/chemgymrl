#useful imports
from typing import NamedTuple, Tuple, Callable, Optional
import gymnasium as gym
import numpy as np

#Imports which need to go soon
import sys
from chemistrylab import vessel
from chemistrylab.reactions.reaction import Reaction
from chemistrylab.benches.characterization_bench import CharacterizationBench
from chemistrylab.util.Visualization import Visualizer

from chemistrylab.lab.shelf import Shelf

Event = vessel.Event

class Action(NamedTuple):
    vessels: Tuple[int]
    parameters: Tuple[tuple]
    event_name: str
    affected_vessels: Optional[Tuple[int]]
    dt: float
    terminal: bool
        
class ContinuousParam(NamedTuple):
    min_val: float
    max_val: float
    thresh: float
    other: tuple

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

    This class represents a bench setup. The setup consists of a shelf with vessels, 
    an action list that describes the actions that can be performed on the vessels by the agent, and 
    a set of events which are performed on the vessels at the end of each step to update it's state 
    (ex. a chemical reaction).

    Args:
        shelf (Shelf): A shelf object to store vessels.
        action_list (list): A list of Action objects containing information describing what each action is.
        observation_list (Tuple[str]): List of observations to obtain from the characteriation bench
        targets (Tuple[str]): The target materials for this bench
        default_events (Tuple[Event]): A list of events to be performed on all working vessels in a bench at the end of each step.
        reward_function (Callable): A function which accepts a target and a list of vessels and outputs a reward.
        discrete (bool): set to True for a discrete action space and False for a continuous one
        max_steps (int): Maximum number of steps for an episode

    """
    def __init__(
        self,
        shelf: Shelf,
        action_list: Tuple[Action],
        observation_list: Tuple[str],
        targets: Tuple[str],
        default_events: Tuple[Event] = tuple(),
        reward_function: Callable = default_reward,
        discrete=True,
        max_steps=50,
    ):
        
                
        # store bench information
        self.shelf=shelf
        self.default_events=default_events
        self.action_list=action_list
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
        self.characterization_bench =  CharacterizationBench(observation_list,self.targets,self.shelf.n_working)
        self.observation_space = gym.spaces.Box(0,1,self.characterization_bench.observation_shape, dtype=np.float32)
        
        # Rendering
        self.render_mode = "rgb_array"
        self.visual = Visualizer(self.characterization_bench)


        self.discrete=discrete
        if self.discrete:
            self.action_space = gym.spaces.Discrete(self.n_actions)
        else:
            self.action_space = gym.spaces.Box(0, 1, (self.n_actions,), dtype=np.float32)
        
        self.disincentive = -0.1

        self.reset()
        
    def get_vessels(self):
        return self.shelf
    def update_vessels(self,new_vessels):
        # TODO: Implement this
        raise NotImplementedError
        
    def build_event(self,action,param):
        """
        Builds a list of (index, :class:`~chemistrylab.vessel.Event`) tuples where index is the index of the vessel the Event will occur in.

        Args:
            action (Action): An action containing information about the vessels and parameters involved in the event.
            param (tuple): A tuple containing parameters for the event.

        Returns:
            List[Tuple]: A list of (index, Event) tuples, where index is the index of the vessel the Event will occur in.
        """
        #watch out for null case
        if action.affected_vessels is None:
            other_vessels = [None]*len(action.vessels)  
        else:
            other_vessels = [self.shelf[i] for i in action.affected_vessels]
        
        #I may change this in the future to remove support for an action to span multiple vessel pairs
        return [(v,Event(action.event_name,param,other_vessels[i])) for i,v in enumerate(action.vessels)]
    
    def _perform_continuous_action(self,action):
        """
        Action should be a 1D array.
        
        Here, event parameters follow the ContinuousParam format. 
        In this implementation all actions are performed with dt=0, then afterwards
        all visible vessels are given an empty event with dt=0.01. 
        (May create a variable to replace 0.01)

        Returns:
            done (bool): Whether or not the episode is done.
            reward (bool): A reward associated with taking the action  (unused).
        """
        for i,(events,_action) in enumerate(self.actions):
            val=action[i]
            #handling any end of episode actions
            if _action.terminal and (val>events[0][1].parameter.thresh):
                return True, 0
            
            #update the involved vessels
            for v,event in events:
                # move the action in [0,1] to [min_val,max_val]
                min_val, max_val, thresh, other = event.parameter
                activ=val*(val>thresh)
                rescaled = activ*(max_val-min_val)+min_val
                #create a new event with this
                param=(rescaled,*other)
                derived_event = Event(event.name,param,event.other_vessel)
                #perform the new event
                self.shelf[v].push_event_to_queue(events=[derived_event], dt=0.0)
        #all vessels which appear in the observation space are updated
        for vessel in self.shelf.get_working_vessels():
            vessel.push_event_to_queue(dt=0.01)
        return False, 0
    def _perform_discrete_action(self,action):
        """
        Action should be an integer.
        Here, the chosen action is perfomed on all relevant vessels, then all
        other visible vessels are fed an empty event with the same dt.

        Returns:
            done (bool): Whether or not the episode is done.
            reward (bool): A reward associated with taking the action (ex if it causes a spill you may get a reward of -0.1).
        """
        reward = 0
        events,_action = self.actions[action]
        updated=set()
        #update the involved vessels
        for v,event in events:
            updated.add(v)
            #going to hard-code dt for now
            feedback = self.shelf[v].push_event_to_queue(events=[event], dt=_action.dt)
            reward += sum(self.disincentive for code in feedback if code==-1)
        #all uninvolved vessels which appear in the observation space
        for v in range(self.shelf.n_working):
            if not v in updated:
                self.shelf[v].push_event_to_queue(dt=_action.dt)
        return _action.terminal,reward
    
    def step(self,action):
        """
        Here, actions are performed by '_perform_discrete_action' or '_perform_continuous_action'. Afterwards, all vessels in
        react_list have their concentrations updated by a reaction (specified in __init__). Finally, a CharacterizationBench
        is used to generate an observation, and a reward function provides a reward (if a terminal state was reached).
        
        Args:
            action (int or 1D array): The action to be performed
        """
        if self.discrete:
            done,reward = self._perform_discrete_action(action)
        else:
            done,reward = self._perform_continuous_action(action)
            
        #perform any default events
        if self.default_events:
          for vessel in self.shelf.get_working_vessels():
            vessel.push_event_to_queue(self.default_events, update_layers=False)
            
        #Increment the step counter and check if you are done
        self.steps+=1
        done=done or self.steps>=self.max_steps
        
        #Handle reward
        if done:
            reward += self.reward_function(self.shelf.get_working_vessels(),self.target_material)-self.initial_reward
        
        #gather observations
        state=self.characterization_bench(self.shelf.get_working_vessels(),self.target_material)
        
        return state, reward, done, False, {}
    
    def _reset(self,target):
        self.steps=0
        self.shelf.reset(target)
        self.target_material=target
        #Rebuild the list of actions with the new vessels
        self.actions = [(self.build_event(a,p),a)  for a in self.action_list for p in a.parameters]
        #Gather the initial reward using a provided reward function
        self.initial_reward = self.reward_function(self.shelf.get_working_vessels(), self.target_material)
        
        return self.characterization_bench(self.shelf.get_working_vessels(), self.target_material)
    
    def reset(self, *args, seed=None, options=None):

        np.random.seed(seed)
        self.action_space.seed(seed)
        target=np.random.choice(self.targets)
        return self._reset(target),{}
        
    def render(self):
        return self.visual.get_rgb(self.shelf.get_working_vessels())