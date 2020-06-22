import numpy as np
import gym
import gym.spaces
import sys

sys.path.append("../../")
from chemistrylab.odeworldgym.odeworld_v0_engine import ODEWorldEnv

class ODEWorldEnv_0(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_0, self).__init__(
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

class ODEWorldEnv_0_Overlap(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_0_Overlap, self).__init__(
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
