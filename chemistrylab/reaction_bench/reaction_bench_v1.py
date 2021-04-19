"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.
"""
# pylint: disable=invalid-name
# pylint: disable=unused-import
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np

import gym
import gym.spaces

sys.path.append("../../")
from chemistrylab.reaction_bench.reaction_bench_v1_engine import ReactionBenchEnv
from chemistrylab.reactions.reaction_base import _Reaction

class ReactionBenchEnv_0(ReactionBenchEnv):
    '''
    Class object to define an environment available in the reaction bench.
    '''

    def __init__(self):
        '''
        Constructor class for the ReactionBenchEnv_0 environment.
        '''

        super(ReactionBenchEnv_0, self).__init__(
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            in_vessel_path=None, # do not include an input vessel
            out_vessel_path=os.getcwd(), # include an output vessel directory
            materials=[ # initialize the bench with the following materials
                {"Material": "1-chlorohexane", "Initial": 1},
                {"Material": "2-chlorohexane", "Initial": 1},
                {"Material": "3-chlorohexane", "Initial": 1},
                {"Material": "Na", "Initial": 1}
            ],
            solutes=[ # initialize the bench with the following solutes available
                {"Material": "H2O", "Initial": 50}
            ],
            n_steps=50,
            dt=0.01,
            overlap=False
        )

class ReactionBenchEnv_1(ReactionBenchEnv):
    '''
    Class object to define an environment available in the reaction bench.
    '''

    def __init__(self):
        '''
        Constructor class for the ReactionBenchEnv_1 environment.
        '''

        super(ReactionBenchEnv_1, self).__init__(
            reaction=_Reaction,
            reaction_file_identifier="decomp",
            in_vessel_path=None, # do not include an input vessel
            out_vessel_path=os.getcwd(), # include an output vessel directory
            materials=[ # initialize the bench with the following materials
                {"Material": "NaCl", "Initial": 0.001},
            ],
            solutes=[ # initialize the bench with the following solutes available
                {"Material": "H2O", "Initial": 0.001}
            ],
            n_steps=50,
            dt=0.01,
            overlap=False
        )

class ReactionBenchEnv_2(ReactionBenchEnv):
    '''
    Class object to define an environment available in the reaction bench.
    '''

    def __init__(self):
        '''
        Constructor class for the ReactionBenchEnv_1 environment.
        '''

        super(ReactionBenchEnv_2, self).__init__(
            reaction=_Reaction,
            reaction_file_identifier="equilibrium",
            in_vessel_path=None, # do not include an input vessel
            out_vessel_path=os.getcwd(), # include an output vessel directory
            materials=[ # initialize the bench with the following materials
                {"Material": "CuSO4", "Initial": 0.001},
                {"Material": "CuS04*5H2O", "Initial": 0.001},
                {"Material": "H2O", "Initial": 0.005}
            ],
            solutes=[ # initialize the bench with the following solutes available
                {"Material": "H2O", "Initial": 0.001}
            ],
            n_steps=50,
            dt=0.01,
            overlap=False
        )

class ReactionBenchEnv_3(ReactionBenchEnv):
    '''
    Class object to define an environment available in the reaction bench.
    '''

    def __init__(self):
        '''
        Constructor class for the ReactionBenchEnv_0 environment.
        '''

        super(ReactionBenchEnv_3, self).__init__(
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz_v1",
            in_vessel_path=None, # do not include an input vessel
            out_vessel_path=os.getcwd(), # include an output vessel directory
            materials=[ # initialize the bench with the following materials
                {"Material": "1-chlorohexane", "Initial": 1},
                {"Material": "2-chlorohexane", "Initial": 1},
                {"Material": "3-chlorohexane", "Initial": 1},
                {"Material": "Na", "Initial": 1}
            ],
            solutes=[ # initialize the bench with the following solutes available
                {"Material": "H2O", "Initial": 50}
            ],
            n_steps=50,
            dt=0.01,
            overlap=False
        )

class ReactionBenchEnv_ODE_Test(ReactionBenchEnv):
    '''
    Class object to define an environment available in the reaction bench.
    '''

    def __init__(self):
        '''
        Constructor class for the ReactionBenchEnv_1 environment.
        '''

        super(ReactionBenchEnv_ODE_Test, self).__init__(
            reaction=_Reaction,
            reaction_file_identifier="ode_check",
            in_vessel_path=None, # do not include an input vessel
            out_vessel_path=os.getcwd(), # include an output vessel directory
            materials=[ # initialize the bench with the following materials
                {"Material": "Na", "Initial": 1},
                {"Material": "Cl", "Initial": 1},
                {"Material": "NaCl", "Initial": 1}
            ],
            solutes=[ # initialize the bench with the following solutes available
                {"Material": "H2O", "Initial": 50}
            ],
            n_steps=50,
            dt=0.01,
            overlap=False
        )

