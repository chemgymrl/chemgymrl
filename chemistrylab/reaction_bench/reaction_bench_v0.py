'''
Module to call the Reaction Bench class and pass parameters to it's engine.

:title: reaction_bench_v0.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-06-23
'''

# pylint: disable=invalid-name
# pylint: disable=unused-import
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np

import gym
import gym.spaces

sys.path.append("../../")
from chemistrylab.reaction_bench.reaction_bench_v0_engine import ReactionBenchEnv
from chemistrylab.reaction_bench.reaction_bench_v0_engine_dissasociation import ReactionBenchEnvDissasociation

class ReactionBenchEnv_0(ReactionBenchEnv):
    '''
    Class object to define an environment available in the reaction bench.
    '''

    def __init__(self):
        '''
        Constructor class for the ReactionBenchEnv_0 environment.
        '''

        super(ReactionBenchEnv_0, self).__init__(
            in_vessel_path=None, # do not include an input vessel
            out_vessel_path=os.getcwd(), # include an output vessel directory
            materials=[ # initialize the bench with the following materials
                {"Material": "1-chlorohexane", "Initial": 0.001},
                {"Material": "2-chlorohexane", "Initial": 0.001},
                {"Material": "3-chlorohexane", "Initial": 0.001},
                {"Material": "Na", "Initial": 0.001}
            ],
            solutes=[ # initialize the bench with the following solutes available
                {"Solute": "H2O", "Initial": 0.001}
            ],
            desired="dodecane",
            overlap=False
        )

class ReactionBenchEnv_0_Overlap(ReactionBenchEnv):
    '''
    Class object to define an environment available in the reaction bench.
    '''

    def __init__(self):
        '''
        Constructor class for the ReactionBenchEnv_0_Overlap environment.
        '''

        super(ReactionBenchEnv_0_Overlap, self).__init__(
            in_vessel_path=None, # do not include an input vessel
            out_vessel_path=os.getcwd(), # include an output vessel directory
            materials=[ # initialize the bench with the following materials
                {"Material": "1-chlorohexane", "Initial": 0.001},
                {"Material": "Na", "Initial": 0.001}
            ],
            solutes=[ # initialize the bench with the following solutes available
                {"Solute": "H2O", "Initial": 0.001}
            ],
            desired="dodecane",
            overlap=True
        )

class ReactionBenchEnv_1(ReactionBenchEnvDissasociation):
    def __init__(self):
        super(ReactionBenchEnv_1, self).__init__(
            materials=[
                {"Material": "NaCl", "Initial": 1},
            ],
            solutes=[
                {"Solute": "H2O", "Initial": 1}
            ],
            desired="Na",
            overlap=False
        )

