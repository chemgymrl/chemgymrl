
import sys
sys.path.append('../../../')

import gymnasium as gym
import time
import pandas as pd
import chemistrylab
import numpy as np
from chemistrylab.chem_algorithms import vessel, material
from copy import deepcopy
from unittest import TestCase


from tests.unit.benches.util import chemgym_filter,check_conservation,check_non_negative,check_conservation_react

ENVS = chemgym_filter([a for a in gym.envs.registry])

def run_env_no_overflow(env_id,seed=1,acts = None):
    """
    Runs a chemgymrl env while making sure no overflow happens in the vessels.
    
    """
    if acts is None:
        acts = []
    env = gym.make(env_id)
    _ = env.reset(seed=seed)
    
    #increase vessel sizes so no overflow happens
    tot_vol = sum(v.filled_volume() for v in env.shelf)
    tot_vol = 2**np.ceil(np.log(tot_vol)/np.log(2))
    for v in env.shelf:
        v.volume=tot_vol
    #collect initial vessels  
    start_vessels = deepcopy([v for v in env.shelf])    
    d = False
    i=0
    while not d:
        if len(acts)<=i:
            act = env.action_space.sample()
            acts+=[act]
        else:
            act=acts[i]
        o,r,d,*_ = env.step(act)
        i+=1
    return start_vessels,env.shelf.get_vessels(),env.reaction_info


class BenchTestCase(TestCase):
    
    def test_nonnegative_distill(self):
        distillations = [a for a in ENVS if "Distill" in a]
        for env_id in distillations:
            for seed in range(1,300,10):
                _,vessels,_ = run_env_no_overflow(env_id,seed)
                self.assertTrue(check_non_negative(vessels))
            #Seed 265 gives a target of NaCl and nothing other than C6H14
            _,vessels,_ = run_env_no_overflow(env_id, 265, acts=[9]*8+[30])
            self.assertTrue(check_non_negative(vessels))
            
    def test_conservation_distill(self):
        distillations = [a for a in ENVS if "Distill" in a]
        for env_id in distillations:
            for seed in range(1,300,10):
                info = run_env_no_overflow(env_id,seed)
                self.assertTrue(check_conservation_react(*info))
                
            info = run_env_no_overflow(env_id, 265, acts=[9]*8+[30])
            self.assertTrue(check_conservation_react(*info))

            
    def test_nonnegative_react(self):
        reactions = [a for a in ENVS if "React" in a]
        for env_id in reactions:
            for seed in range(1,300,10):
                _,vessels,_ = run_env_no_overflow(env_id,seed)
                self.assertTrue(check_non_negative(vessels))
            
    def test_conservation_react(self):
        reactions = [a for a in ENVS if "React" in a]
        for env_id in reactions:
            for seed in range(1,300,10):
                info = run_env_no_overflow(env_id,seed)
                self.assertTrue(check_conservation_react(*info))
            
    def test_nonnegative_extract(self):
        extractions = [a for a in ENVS if "Extract" in a]
        for env_id in extractions:
            for seed in range(1,300,10):
                _,vessels,_ = run_env_no_overflow(env_id,seed)
                self.assertTrue(check_non_negative(vessels))
            
    def test_conservation_extract(self):
        extractions = [a for a in ENVS if "Extract" in a]
        for env_id in extractions:
            for seed in range(1,300,10):
                v_start,v_end,react_info = run_env_no_overflow(env_id,seed)
                self.assertTrue(check_conservation(v_start,v_end))            
