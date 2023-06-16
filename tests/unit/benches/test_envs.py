def null_test(x):
    raise NotImplementedError

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

from tests.unit.benches.util import chemgym_filter,check_conservation,check_non_negative,check_conservation_react

ENVS = chemgym_filter([a for a in gym.envs.registry])


def get_test_names(env_id):
    env=gym.make(env_id)
    names = set()
    for action in env.action_list:
        name = "test_"+env_id.replace("-", "_") +"_"+ action.event_name.replace(" ", "_")
        names.add(name)
    return names

def null_test(x):
    raise NotImplementedError

class EnvTestCase(TestCase):
    pass


# The below code adds a test case for each type of action of each bench to the EnvTestCase class which raises a NotImplementedError
# (it is only added if a test of that name doesn't exist yet)
testnames = [t for env_id in ENVS for t in get_test_names(env_id)]
method_list = [attribute for attribute in dir(EnvTestCase) if callable(getattr(EnvTestCase, attribute))]
for testname in testnames:
    if not testname in method_list:
        setattr(EnvTestCase,testname,null_test)
    
    