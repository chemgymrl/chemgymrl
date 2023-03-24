from gym import spaces
import gym
import numpy as np
import importlib


class DiscreteWrapper(gym.Env):
    """
    Wrapper to Turn MultiDiscrete and Continuous ChemGym Environments into Discrete environments
    """
    def __init__(self,entry_point,exclude=0,nbins=5,null_act=0):
        """
        Parameters
        ---------------
            `entry_point` : `str` 
            Import path to and name of your environment class, formatted the same as when registering a gym
            `exclude`     : `int`
            Number of redundant actions (assumed to be at the end)
            `nbins`       : `int`
            Number of bins (per dimension) when converting to a continuous action space
            `null_act`    : `tuple`
            Values in the continuous action space which correspond to doing nothing
        """
        mod_name, attr_name = entry_point.split(":")
        #Build the MultiDiscrete gym
        mod = importlib.import_module(mod_name)
        self.gym = getattr(mod, attr_name)()
        
        if type(self.gym.action_space) is gym.spaces.MultiDiscrete:
            #This will only work with multidiscrete shape (2,) action spaces
            self.x,self.y = self.gym.action_space.nvec
            #Translate the action space to a single Discrete
            self.action_space=spaces.Discrete(self.x*self.y-exclude)
            #create table mapping single discrete actions to multidiscrete actions
            self.actions=np.zeros([self.x*self.y-exclude,2],dtype=np.int32)
            for i in range(self.x*self.y-exclude):
                self.actions[i] = np.array([i//self.y,i%self.y])
        elif type(self.gym.action_space) is gym.spaces.Box:
            
            self.actions=np.zeros([self.gym.action_space.shape[0]*nbins+1,self.gym.action_space.shape[0]],dtype=np.int32)
            # Get the action that corresponds to doing nothing and make that action zero
            self.actions[0]=null_act
            #set the number of possible actions
            self.action_space=spaces.Discrete(self.actions.shape[0])
            
            #Discretize the continuous actions by only moving in cardinal directions
            for i in range(self.gym.action_space.shape[0]):
                # get the range in this direction
                low = self.gym.action_space.low[i]
                high = self.gym.action_space.high[i]
                for j in range(nbins):
                    self.actions[i*nbins+j+1] = null_act
                    #scale the action linearly
                    self.actions[i*nbins+j+1][i] = (j/(nbins-1))*(high-low)+low
                    
        self.observation_space=self.gym.observation_space
    
    def step(self,act):
        #create the multidiscrete input with your single number
        step_results = self.gym.step(self.actions[act])
        self.get_env_vessels()
        return step_results
    
    def reset(self):
        reset_results = self.gym.reset()
        self.get_env_vessels()
        return reset_results
    
    def update_vessel(self, new_vessel):
        update_vessel_results = self.gym.update_vessel(new_vessel)
        self.get_env_vessels()
        return update_vessel_results
    
    def get_env_vessels(self):
        self.vessels = self.gym.vessels