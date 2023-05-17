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
sys.path.append("../../")
from chemistrylab.chem_algorithms.reward import RewardGenerator
from chemistrylab.chem_algorithms import material, vessel
from chemistrylab.benches.general_bench import *
import importlib
from chemistrylab.reactions.reaction_info import ReactInfo, REACTION_PATH
from chemistrylab.lab.shelf import Shelf

def get_mat(mat,amount,name=None):
    "Makes a Vessel with a single material"
    
    my_vessel = vessel.Vessel(
        label=f'{mat} Vessel' if name is None else name,
        ignore_layout=True
    )
    # create the material dictionary for the vessel
    matclass = material.REGISTRY[mat]()
    matclass.mol=amount
    material_dict = {mat:matclass}
    # instruct the vessel to update its material dictionary

    my_vessel.material_dict=material_dict
    my_vessel.validate_solvents()
    my_vessel.validate_solutes()
    my_vessel.default_dt=0.01
    
    return my_vessel

        
class GeneralWurtzReact_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,include_dissolved=False)
        shelf = Shelf([
            get_mat("diethyl ether",4,"Reaction Vessel"),
            get_mat("1-chlorohexane",1),
            get_mat("2-chlorohexane",1),
            get_mat("3-chlorohexane",1),
            get_mat("Na",3),
        ])
        actions = [
            Action([0],    [ContinuousParam(156,307,0,500)],   'heat contact',   [0],  0.01,  False),
            Action([1],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],  0.01,  False),
            Action([2],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],  0.01,  False),
            Action([3],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],  0.01,  False),
            Action([4],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],  0.01,  False),
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json")
        
        super(GeneralWurtzReact_v2, self).__init__(
            shelf,
            actions,
            react_info,
            ["PVT","spectra","targets"],
            reward_function=r_rew,
            react_list=[0],
            targets=None,
            discrete=False,
            max_steps=20
        )
        
class GeneralWurtzReact_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,include_dissolved=False)
        shelf = Shelf([
            get_mat("diethyl ether",4,"Reaction Vessel"),
            get_mat("1-chlorohexane",1),
            get_mat("2-chlorohexane",1),
            get_mat("3-chlorohexane",1),
            get_mat("Na",3),
        ])
        actions = [
            Action([0],    [ContinuousParam(156,307,0,500)],   'heat contact',   [0],  0.01,  False),
            Action([1],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],  0.01,  False),
            Action([2],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],  0.01,  False),
            Action([3],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],  0.01,  False),
            Action([4],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],  0.01,  False),
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json")
        
        super(GeneralWurtzReact_v0, self).__init__(
            shelf,
            actions,
            react_info,
            ["PVT","spectra","targets"],
            reward_function=r_rew,
            react_list=[0],
            targets=react_info.PRODUCTS[:-1],
            discrete=False,
            max_steps=20
        )

class FictReact_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,
                                include_dissolved=False, exclude_mat = "fict_E")
        shelf = Shelf([
            get_mat("H2O",30,"Reaction Vessel"),
            get_mat("fict_A",1),
            get_mat("fict_B",1),
            get_mat("fict_C",1),
            get_mat("fict_D",3),
        ])

        actions = [
            Action([0],    [ContinuousParam(273,373,0,300)],   'heat contact',    [0],   0.01,   False),
            Action([1],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],   0.01,   False),
            Action([3],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],   0.01,   False),
            Action([4],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],   0.01,   False),
        ]
        
        targets = ["fict_E", "fict_F", "fict_G", "fict_H", "fict_I"]
        react_info = ReactInfo.from_json(REACTION_PATH+"/fict_react.json")

        super(FictReact_v2, self).__init__(
            shelf,
            actions,
            react_info,
            ["PVT","spectra","targets"],
            reward_function=r_rew,
            react_list=[0],
            targets=targets,
            discrete=False,
            max_steps=20
        )
        

class FictReactBandit_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self,targets=None):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,
                                include_dissolved=False, exclude_mat = "fict_E")
        shelf = Shelf([
            get_mat("H2O",30,"Reaction Vessel"),
            get_mat("fict_A",1),
            get_mat("fict_B",1),
            get_mat("fict_C",1),
            get_mat("fict_D",3),
        ])

        actions = [
            Action([0],    [ContinuousParam(273,373,0,300)],   'heat contact',    [0],   0.01,   False),
            Action([1],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],   0.01,   False),
            Action([3],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],   0.01,   False),
            Action([4],    [ContinuousParam(0,1,1e-3,None)],   'pour by percent',  [0],   0.01,   False),
        ]
        if targets is None:
            targets = ["fict_E", "fict_F", "fict_G", "fict_H", "fict_I"]
        react_info = ReactInfo.from_json(REACTION_PATH+"/fict_react.json")

        super().__init__(
            shelf,
            actions,
            react_info,
            ["targets"],
            reward_function=r_rew,
            react_list=[0],
            targets=targets,
            discrete=False,
            max_steps=20
        )
        self.action_space = gym.spaces.Box(0, 1, (self.n_actions+4,), dtype=np.float32)

    def step(self,action):
        action=np.array(action)
        uaction = action[:-4]
        gate = action[-4:]*self.max_steps-0.5
        ret=0
        d=False
        while not d:
            act = uaction*1
            act[1:]*= (gate<self.steps)
            o,r,d,*_ = super().step(act)
            gate[gate<self.steps-1]=self.max_steps
            ret+=r
        return o,ret,d,*_