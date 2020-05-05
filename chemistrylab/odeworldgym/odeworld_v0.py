import numpy as np
import gym
import gym.spaces
from chemistrylab.odeworldgym.odeworld_v0_engine import ODEWorldEnv
import sys
from chemistrylab.reactions.reaction_0 import Reaction as Reaction_0
from chemistrylab.reactions.reaction_1 import Reaction as Reaction_1
from chemistrylab.reactions.reaction_2 import Reaction as Reaction_2
from chemistrylab.reactions.reaction_3 import Reaction as Reaction_3
from chemistrylab.reactions.reaction_4 import Reaction as Reaction_4
from chemistrylab.reactions.reaction_5 import Reaction as Reaction_5
from chemistrylab.reactions.reaction_6 import Reaction as Reaction_6

class ODEWorldEnv_0(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_0, self).__init__(
            reaction = Reaction_0,
        )

class ODEWorldEnv_0_Overlap(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_0_Overlap, self).__init__(
            reaction = Reaction_0,
            overlap = True,
        )

class ODEWorldEnv_1(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_1, self).__init__(
            reaction = Reaction_1
        )

class ODEWorldEnv_1_Overlap(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_1_Overlap, self).__init__(
            reaction = Reaction_1,
            overlap = True,
        )

class ODEWorldEnv_2(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_2, self).__init__(
            reaction = Reaction_2
        )

class ODEWorldEnv_2_Overlap(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_2_Overlap, self).__init__(
            reaction = Reaction_2,
            overlap = True,
        )

class ODEWorldEnv_3(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_3, self).__init__(
            reaction = Reaction_3
        )

class ODEWorldEnv_3_Overlap(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_3_Overlap, self).__init__(
            reaction = Reaction_3,
            overlap = True,
        )

class ODEWorldEnv_4(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_4, self).__init__(
            reaction = Reaction_4
        )

class ODEWorldEnv_4_Overlap(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_4_Overlap, self).__init__(
            reaction = Reaction_4,
            overlap = True,
        )

class ODEWorldEnv_5(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_5, self).__init__(
            reaction = Reaction_5
        )

class ODEWorldEnv_5_Overlap(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_5_Overlap, self).__init__(
            reaction = Reaction_5,
            overlap = True,
        )

class ODEWorldEnv_6(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_6, self).__init__(
            reaction = Reaction_6
        )

class ODEWorldEnv_6_Overlap(ODEWorldEnv):
    def __init__(self):
        super(ODEWorldEnv_6_Overlap, self).__init__(
            reaction = Reaction_6,
            overlap = True,
        )
