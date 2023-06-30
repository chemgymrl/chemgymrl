

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

# import external modules
import os
import pickle
import sys
from copy import deepcopy

from chemistrylab.util.reward import RewardGenerator
from chemistrylab import material, vessel
from chemistrylab.benches.general_bench import *
from chemistrylab.reactions.reaction_info import ReactInfo, REACTION_PATH
import importlib
from chemistrylab.lab.shelf import VariableShelf

def wurtz_vessel(add_mat):
    """
    Function to generate an input vessel for the wurtz distillation experiment.

    Args:
        add_mat (str): The target material to include in the vessel

    Returns:
        Vessel: A vessel containing add_mat and some undesired materials
    """

    # initialize extraction vessel
    boil_vessel = vessel.Vessel(label='boil_vessel')

    # initialize C6H14
    C6H14 = material.C6H14()
    C6H14.mol = 4.0

    # initialize material
    products = {'dodecane': material.Dodecane,
        '5-methylundecane': material.FiveMethylundecane,
        '4-ethyldecane': material.FourEthyldecane,
        '5,6-dimethyldecane': material.FiveSixDimethyldecane,
        '4-ethyl-5-methylnonane': material.FourEthylFiveMethylnonane,
        '4,5-diethyloctane': material.FourFiveDiethyloctane,
        'NaCl': material.NaCl
    }

    try:
        if add_mat == "":
            add_mat = choice(list(products.keys()))
        
        add_material = products[add_mat]()
    
    except KeyError:
        add_mat = 'dodecane'
        add_material = products[add_mat]()
    
    if (add_mat != 'NaCl'):
        add_material.set_solute_flag(True)
        #add_material.set_color(0.0)
        add_material.phase = 'l'
    add_material.mol=1

    # material_dict
    material_dict = {
        C6H14.get_name(): C6H14,
        add_material.get_name(): add_material
    }

    if np.random.choice([0, 1]) > 0.5:
        add_material2 = material.Dodecane() if (add_mat == 'NaCl') else material.NaCl()
        add_material2.mol=1
        material_dict[add_material2.get_name()] = add_material2


    boil_vessel.material_dict=material_dict

    boil_vessel.validate_solvents()
    boil_vessel.validate_solutes()

    boil_vessel.default_dt=0.01
    
    return boil_vessel, add_mat




class GeneralWurtzDistill_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    def __init__(self):
        d_rew= RewardGenerator(use_purity=True,exclude_solvents=False,include_dissolved=True)
        shelf = VariableShelf( [
            lambda x:wurtz_vessel(x)[0],
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Beaker 2"),
        ],[], n_working = 3)

        amounts=np.linspace(0,1,10).reshape([10,1])
        
        heat_info = np.zeros([10,2])
        #Water heat transfer
        heat_info[:5,1]=200.0
        # Water temperatures
        heat_info[:5,0] = np.linspace(270,300,5)
        #Bunsen Burner heat transfer
        heat_info[5:,1]=30.0
        # Bunsen Burner Temperatures
        heat_info[5:,0] = np.linspace(370,1000,5)

        actions = [
            Action([0],    heat_info,           'heat contact',    [1],   0.01,   False),
            Action([0],    amounts,              'pour by volume', [1],   0.01,   False),
            Action([1],    amounts,              'pour by volume', [2],   0.01,   False),
            #Action([0], [[T, False], [T, True]], 'wait',           [1],   0.01,   False),
            Action([0],  [[0]],                  'mix',            None,  0,      True)
        ]
        
        targets = [
            "dodecane",
            "5-methylundecane",
            "4-ethyldecane",
            "5,6-dimethyldecane",
            "4-ethyl-5-methylnonane",
            "4,5-diethyloctane",
            "NaCl"
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/precipitation.json")

        reaction = Reaction(react_info)
        reaction.solver="newton"
        reaction.newton_steps=100
        
        super(GeneralWurtzDistill_v2, self).__init__(
            shelf,
            actions,
            ["layers","PVT","targets"],
            targets=targets,
            default_events = (Event("react", (reaction,), None),),
            reward_function=d_rew,
        )




class WurtzDistillDemo_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }

    def __init__(self):
        d_rew= RewardGenerator(use_purity=True,exclude_solvents=False,include_dissolved=True)
        shelf = VariableShelf( [
            lambda x:wurtz_vessel(x)[0],
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Beaker 2"),
        ],[], n_working = 3)

        amounts=np.ones([1,1])*0.02
        
        heat_info = np.array([
            [300,20], # Room temp water
            [270,20], # Freezing water
            [1000,3]  # Fire
        ])

        actions = [
            Action([0],    heat_info,            'heat contact',   [1],   0.01,   False),
            Action([0],    amounts,              'pour by volume', [1],   0.01,   False),
            Action([1],    amounts,              'pour by volume', [2],   0.01,   False),
            Action([],     [[]],                 'End',            None,  0,      True)
        ]
        
        targets = [
            "dodecane",
            "5-methylundecane",
            "4-ethyldecane",
            "5,6-dimethyldecane",
            "4-ethyl-5-methylnonane",
            "4,5-diethyloctane",
            "NaCl"
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/precipitation.json")
        reaction = Reaction(react_info)
        reaction.solver="newton"
        reaction.newton_steps=100

        super(WurtzDistillDemo_v0, self).__init__(
            shelf,
            actions,
            ["layers","PVT","targets"],
            targets=targets,
            default_events = (Event("react", (reaction,), None),),
            reward_function=d_rew,
            max_steps=500
        )



    def get_keys_to_action(self):
        # Control with the numpad or number keys.
        keys = {(ord(k),):i for i,k in enumerate("123456") }
        keys[()]=0
        return keys