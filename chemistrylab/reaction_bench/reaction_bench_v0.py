'''
Module to call the Reaction Bench class and pass parameters to it's engine.

:title: reaction_bench_v0.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-06-23
'''

import numpy as np
import gym
import gym.spaces
import sys

sys.path.append("../../")
from chemistrylab.reaction_bench.reaction_bench_v0_engine import ReactionBenchEnv

class ReactionBenchEnv_0(ReactionBenchEnv):
    def __init__(self):
        super(ReactionBenchEnv_0, self).__init__(
            materials=[
                {"Material": "1-chlorohexane", "Initial": 1},
                {"Material": "2-chlorohexane", "Initial": 1},
                {"Material": "3-chlorohexane", "Initial": 1},
                {"Material": "Na", "Initial": 1}
            ],
            solutes=[
                {"Solute": "ethoxyethane", "Initial": 1}
            ],
            desired="dodecane",
            overlap=False
        )

class ReactionBenchEnv_0_Overlap(ReactionBenchEnv):
    def __init__(self):
        super(ReactionBenchEnv_0_Overlap, self).__init__(
            materials=[
                {"Material": "1-chlorohexane", "Initial": 1},
                {"Material": "Na", "Initial": 1}
            ],
            solutes=[
                {"Solute": "ethoxyethane", "Initial": 1}
            ],
            desired="dodecane",
            overlap=True
        )
