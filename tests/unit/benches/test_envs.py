import sys
sys.path.append('../../../')

import gymnasium as gym
import time
import pandas as pd
import chemistrylab
import numpy as np
from chemistrylab import vessel, material
from copy import deepcopy
from unittest import TestCase

from tests.unit.benches.util import chemgym_filter,check_conservation,check_non_negative,check_conservation_react, run_env_no_overflow, prep_no_overflow

ENVS = chemgym_filter([a for a in gym.envs.registry])


def get_test_names(env_id):
    env=gym.make(env_id)
    names = set()
    for action in env.action_list:
        name = "test_"+env_id.replace("-", "_") +"_"+ action.event_name.replace(" ", "_")
        names.add(name)
    names.add("test_"+env_id.replace("-", "_") +"_observation")
    return names

def last_event(action,shelf, param = None):
    """Grab the event with the last parameter and vessel config from an Action object"""
    if param is None:
        param = action.parameters[-1]
    if action.affected_vessels is None:
        other_vessels = [None]*len(action.vessels)  
    else:
        other_vessels = [shelf[i] for i in action.affected_vessels]

    #I may change this in the future to remove support for an action to span multiple vessel pairs
    return [(v,vessel.Event(action.event_name,param,other_vessels[i])) for i,v in enumerate(action.vessels)][-1]


def null_test(x):
    raise NotImplementedError

class EnvTestCase(TestCase):
    def test_FictReact_v2_heat_contact(self):
        env = gym.make("FictReact-v2")
        env.reset()
        t0 = env.shelf[0].temperature
        t_increase = vessel.Event("heat contact",(1000,1),env.shelf[0])
        env.shelf[0].push_event_to_queue([t_increase])
        t1 = env.shelf[0].temperature
        self.assertTrue(t1>t0)
        t_decrease = vessel.Event("heat contact",(100,1),env.shelf[0])
        env.shelf[0].push_event_to_queue([t_decrease])
        t2 = env.shelf[0].temperature
        self.assertTrue(t1>t2)
        env.step([1,0,0,0,0])
        t3 = env.shelf[0].temperature
        self.assertTrue(t3>t2)
        env.step([0,0,0,0,0])
        t4 = env.shelf[0].temperature
        self.assertTrue(t3>t4)
    def test_FictReact_v2_observation(self):
        env = gym.make("FictReact-v2")
        env.reset()
        o,r,d,*_ = env.step([1,1,1,1,1])
        self.assertTrue(np.max(o)<=1 and np.min(o)>=0)
        #TODO: Finish observation test
    def test_FictReact_v2_pour_by_percent(self):
        #conservation of stuff works
        info = run_env_no_overflow("FictReact-v2", 10, acts=[[1,1,1,1,1]]+[[1,0,0,0,0]]*19)
        self.assertTrue(check_conservation_react(*info))
        env = gym.make("FictReact-v2")
        env.reset()
        #pouring reduces volume of source vessel
        v0 = env.shelf[1].filled_volume()
        env.step([0,1,0,0,0])
        v1 = env.shelf[1].filled_volume()
        self.assertTrue(v1<v0)
        
    def test_GenWurtzDistill_v2_heat_contact(self):
        env = gym.make("GenWurtzDistill-v2")
        _=env.reset()
        mdict = env.shelf[0].material_dict
        #grab all present boiling points
        bp = sorted([(key,mdict[key]._boiling_point) for key in mdict],key = lambda x:x[1])
        i=0
        #boil off all of the materials
        while i<len(bp):
            event = vessel.Event("heat contact",(bp[i][1]+10,50),env.shelf[1])
            env.shelf[0].push_event_to_queue([event])
            remaining = env.shelf[0].material_dict[bp[i][0]].mol
            #make sure the temperature is below or equal to the lowest boiling point
            self.assertTrue( remaining<1e-12 or env.shelf[0].temperature<=bp[i][1] )
            if remaining<1e-12:
                i+=1
        #Make sure boiled off materials have nonzero mols
        self.assertTrue(all([env.shelf[0].material_dict[pair[0]].mol>=0 for pair in bp]))
        t=env.shelf[0].temperature
        # testing cooling to 273K
        for x in range(100):
            t0=t
            event = vessel.Event("heat contact",(273,100),env.shelf[1])
            env.shelf[0].push_event_to_queue([event])
            t=env.shelf[0].temperature
            self.assertTrue(abs(t-273)<1e7 or t>273 and t<t0)
        # testing heating to 500K
        for x in range(100):
            t0=t
            event = vessel.Event("heat contact",(500,100),env.shelf[1])
            env.shelf[0].push_event_to_queue([event])
            t=env.shelf[0].temperature
            self.assertTrue(abs(t-500)<1e7 or t<500 and t>t0)
            
    def test_GenWurtzDistill_v2_pour_by_volume(self):
        env = gym.make("GenWurtzDistill-v2")
        _=env.reset()

        #get pouring events
        pour_1 = last_event(env.action_list[1],env.shelf,(0.5,))
        pour_2 = last_event(env.action_list[2],env.shelf,(0.5,))

        #test pouring shelf[0]->shelf[1]
        src_0 = env.shelf[0].filled_volume()
        dest_0 = env.shelf[1].filled_volume()
        env.shelf[pour_1[0]].push_event_to_queue(pour_1[1:])

        src_1 = env.shelf[0].filled_volume()
        dest_1 = env.shelf[1].filled_volume() 

        #make sure source vessel volume decreases
        self.assertTrue( src_1<src_0)
        #make sure destination vessel volume increases
        self.assertTrue(dest_1>dest_0)

        for x in range(4):
            env.shelf[pour_1[0]].push_event_to_queue(pour_1[1:])
        check_non_negative(env.shelf)
        
        #test pouring shelf[1]->shelf[2]
        src_2 = env.shelf[1].filled_volume()
        dest_2 = env.shelf[2].filled_volume()
        
        env.shelf[pour_2[0]].push_event_to_queue(pour_2[1:])
        
        src_3 = env.shelf[0].filled_volume()
        dest_3 = env.shelf[1].filled_volume() 
        
        #make sure source vessel volume decreases
        self.assertTrue( src_3<src_2)
        #make sure destination vessel volume increases
        self.assertTrue(dest_3>dest_2)
        
        for x in range(4):
            env.shelf[pour_2[0]].push_event_to_queue(pour_2[1:])
        check_non_negative(env.shelf)
        
    def test_GenWurtzDistill_v2_mix(self):
        env = gym.make("GenWurtzDistill-v2")
        _=env.reset()
        
        o,r,done,*_ = env.step(30)
        
        self.assertTrue(done)
        
    def test_GenWurtzDistill_v2_observation(self):
        env = gym.make("GenWurtzDistill-v2")
        _=env.reset()
        o,r,d,*_ = env.step(0)
        self.assertTrue(np.max(o)<=1 and np.min(o)>=0)
        
    def test_GenWurtzExtract_v2_drain_by_pixel(self):
        env = gym.make("GenWurtzExtract-v2")
        _=env.reset(seed=10)
        
        src_0 = env.shelf[0].filled_volume()
        dest_0 = env.shelf[1].filled_volume()
        
        #settle the ev
        env.step(39)
        # Draining action for 2 pixels
        o,r,d,*_ = env.step(0)
        
        
        src_1 = env.shelf[0].filled_volume()
        dest_1 = env.shelf[1].filled_volume()
        
        #make sure source vessel volume decreases
        self.assertTrue( src_1<src_0)
        #make sure destination vessel volume increases
        self.assertTrue(dest_1>dest_0)
        
        #checking nonnegative
        for x in range(12):
            #max pouring
            env.step(9)
        self.assertTrue(env.shelf[0].filled_volume()>=0)
        self.assertTrue(check_non_negative(env.shelf))
        
        #checking conservation
        _=env.reset()
        
        initial = prep_no_overflow(env)
        
        for x in range(4):
            #max pouring
            env.step(9)
        
        self.assertTrue(check_conservation(initial,env.shelf))
        
        
# The below code adds a test case for each type of action of each bench to the EnvTestCase class which raises a NotImplementedError
# (it is only added if a test of that name doesn't exist yet)
testnames = [t for env_id in ENVS for t in get_test_names(env_id)]
method_list = [attribute for attribute in dir(EnvTestCase) if callable(getattr(EnvTestCase, attribute))]
for testname in testnames:
    if not testname in method_list:
        setattr(EnvTestCase,testname,null_test)
    
    