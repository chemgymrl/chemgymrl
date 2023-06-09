# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

# import external modules
import os
import pickle
import sys
from random import choice
from copy import deepcopy

from chemistrylab.util.reward import RewardGenerator
from chemistrylab import material, vessel
from chemistrylab.benches.general_bench import *

import importlib

from chemistrylab.reactions.reaction_info import ReactInfo, REACTION_PATH
from chemistrylab.lab.shelf import Shelf,VariableShelf


def wurtz_vessel(add_mat=""):
    """
    Function to generate an input vessel for the wurtz extraction bench.

    Args:
    - add_mat (str): The target material to include in the vessel

    Returns:
    - extract_vessel (Vessel): A vessel containing add_mat and some undesired materials
    """

    # initialize extraction vessel
    extraction_vessel = vessel.Vessel(label='extraction_vessel')

    # initialize DiEthylEther
    DiEthylEther = material.DiEthylEther()

    # initialize Na
    Na = material.Na()
    Na.charge = 1.0
    Na.set_solute_flag(True)
    Na.set_color(0.0)
    Na.polarity = 2.0
    Na.phase = 'l'
    Na._boiling_point = material.NaCl()._boiling_point

    # initialize Cl
    Cl = material.Cl()
    Cl.charge = -1.0
    Cl.set_solute_flag(True)
    Cl.set_color(0.0)
    Cl.polarity = 2.0
    Cl.phase = 'l'
    Cl._boiling_point = material.NaCl()._boiling_point

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
        
        if add_mat != "NaCl":
            add_material = products[add_mat]()
        else:
            add_material = products['dodecane']()
    
    except KeyError:
        add_mat = 'dodecane'
        add_material = products['dodecane']()
    
    add_material.set_solute_flag(True)
    add_material.set_color(0.0)
    add_material.phase = 'l'


    DiEthylEther.mol=4.0
    Na.mol=1.0
    Cl.mol=1.0
    add_material.mol=1.0
    # material_dict
    material_dict = {
        DiEthylEther.get_name(): DiEthylEther,
        Na.get_name(): Na,
        Cl.get_name(): Cl,
        add_material.get_name(): add_material,
    }

    extraction_vessel.material_dict=material_dict

    extraction_vessel.validate_solvents()
    extraction_vessel.validate_solutes()

    extraction_vessel._mix(-1000,None,-1000)

    return extraction_vessel, add_mat


def oil_vessel():
    """
    Function to generate an input vessel for the oil extraction bench.

    Returns:
    - extract_vessel (Vessel): A vessel containing oil and NaCl
    """
    # initialize extraction vessel
    extraction_vessel = vessel.Vessel(label='extraction_vessel')
    # initialize H2O
    C6H14 = material.C6H14()
    C6H14.mol=1.0
    # Get dissolved NaCl
    dissolved = material.NaCl().dissolve()
    for d in dissolved:
        d.mol=dissolved[d]
    mats = [C6H14]+[d for d in dissolved]

    # material_dict
    material_dict = {mat._name:mat for mat in mats}
    # Set up the vessel
    extraction_vessel.material_dict=material_dict
    extraction_vessel.validate_solvents()
    extraction_vessel.validate_solutes()

    return extraction_vessel


def make_solvent(mat):
    "Makes a Vessel with a single material"
    solvent_vessel = vessel.Vessel(
        label=f'{mat} Vessel',
    )
    # create the material dictionary for the solvent vessel
    solvent_class = material.REGISTRY[mat]()
    solvent_class.set_solvent_flag(True)
    solvent_class.mol=1e6
    solvent_vessel.material_dict = {mat:solvent_class}
    # instruct the vessel to update its material dictionary
    return solvent_vessel


class GeneralWurtzExtract_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    def __init__(self):
        e_rew= RewardGenerator(use_purity=True,exclude_solvents=True,include_dissolved=True)
        shelf = VariableShelf( [
            lambda x:wurtz_vessel(x)[0],
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Beaker 2"),
            lambda x:make_solvent("C6H14"),
            lambda x:make_solvent("diethyl ether")
        ],[], n_working = 3)
        amounts=np.linspace(0.2,1,5).reshape([5,1])
        pixels = (amounts*10).astype(np.int32)
        actions = [
            Action([0], pixels,              'drain by pixel',[1],  0.01, False),
            Action([0],-amounts,             'mix',           None, 0.01, False),
            Action([1], amounts,             'pour by volume',[0],  0.01, False),
            Action([2], amounts,             'pour by volume',[0],  0.01, False),
            Action([0], amounts,             'pour by volume',[2],  0.01, False),
            #If pouring by volume takes time, then there is no change in observation when waiting after pouring in some cases
            Action([3], amounts/2,           'pour by volume',[0],  0,    False),
            Action([4], amounts/2,           'pour by volume',[0],  0,    False),
            Action([0,1,2], 32**amounts/200, 'mix',           None, 0,    False),
            Action([0], [[0]],               'mix',           None, 0,    True)
        ]
        
        targets = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json").PRODUCTS

        super(GeneralWurtzExtract_v2, self).__init__(
            shelf,
            actions,
            ["layers","targets"],
            targets,
            reward_function=e_rew
        )


class WaterOilExtract_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    def __init__(self):
        e_rew= RewardGenerator(use_purity=False, exclude_solvents=True, include_dissolved=True, exclude_mat="C6H14")
        shelf =VariableShelf( [
            lambda x:oil_vessel(),
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Waste Vessel"),
            lambda x:make_solvent("C6H14"),
            lambda x:make_solvent("H2O")
        ], [], n_working = 2)
        amounts=np.linspace(0.2,1,5).reshape([5,1])
        pixels = (amounts*10).astype(np.int32)
        actions = [
            Action([0], pixels,              'drain by pixel',[1],  0.01, False),
            Action([0],-amounts,             'mix',           None, 0.01, False),
            Action([1], amounts,             'pour by volume',[0],  0.01, False),
            Action([2], amounts,             'pour by volume',[0],  0.01, False),
            Action([0], amounts,             'pour by volume',[2],  0.01, False),
            #If pouring by volume takes time, then there is no change in observation when waiting after pouring in some cases
            Action([3], amounts/2,           'pour by volume',[0],  0,    False),
            Action([4], amounts/2,           'pour by volume',[0],  0,    False),
            Action([0,1,2], 32**amounts/200, 'mix',           None, 0,    False),
            Action([0], [[0]],               'mix',           None, 0,    True)
        ]
        

        super(WaterOilExtract_v0, self).__init__(
            shelf,
            actions,
            ["layers","targets"],
            targets=["NaCl"],
            reward_function=e_rew,
        )




class WurtzExtractDemo_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }


    def __init__(self):
        e_rew= RewardGenerator(use_purity=True,exclude_solvents=True,include_dissolved=True)
        shelf = VariableShelf( [
            lambda x:wurtz_vessel(x)[0],
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Beaker 2"),
            lambda x:make_solvent("C6H14"),
            lambda x:make_solvent("diethyl ether")
        ],[], n_working = 3)
        amounts=np.ones([1,1])*0.02
        pixels = [[1]]
        actions = [
            Action([0], pixels,              'drain by pixel',[1],  0.001, False),
            Action([0],-amounts,             'mix',           None, 0,     False),
            Action([1], amounts,             'pour by volume',[0],  0.001, False),
            Action([2], amounts,             'pour by volume',[0],  0.001, False),
            Action([0], amounts,             'pour by volume',[2],  0.001, False),
            Action([3], amounts/2,           'pour by volume',[0],  0,    False),
            Action([4], amounts/2,           'pour by volume',[0],  0,    False),
            Action([0,1,2], [[1e-3],[0.016]],'mix',           None, 0,    False),
            Action([0], [[0]],               'mix',           None, 0,    True)
        ]
        
        targets = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json").PRODUCTS

        super(WurtzExtractDemo_v0, self).__init__(
            shelf,
            actions,
            ["layers","targets"],
            targets,
            reward_function=e_rew,
            max_steps=500
        )

    def get_keys_to_action(self):
        # Control with the numpad or number keys.
        keys = {(ord(k),):i for i,k in enumerate("1234567890") }
        keys[()]=7
        return keys

class SeparateTest_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }
    
    def __init__(self):
        e_rew= RewardGenerator(use_purity=True,exclude_solvents=True,include_dissolved=True)
        shelf = VariableShelf( [
            lambda x:wurtz_vessel(x)[0],
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Beaker 2"),
            lambda x:make_solvent("C6H14"),
            lambda x:make_solvent("diethyl ether"),
            lambda x:make_solvent("H2O"),
            lambda x:make_solvent("ethoxyethane"),
            lambda x:make_solvent("ethyl acetate"),
        ],[], n_working = 3)
        amounts=np.ones([1,1])*0.02
        pixels = [[1]]
        actions = [
            Action([0], pixels,              'drain by pixel',[1],  0.001, False),
            Action([0],-amounts,             'mix',           None, 0,     False),
            Action([1], amounts,             'pour by volume',[0],  0.001, False),
            Action([2], amounts,             'pour by volume',[0],  0.001, False),
            Action([0], amounts,             'pour by volume',[2],  0.001, False),
            Action([3], amounts/2,           'pour by volume',[0],  0,    False),
            Action([4], amounts/2,           'pour by volume',[0],  0,    False),
            Action([5], amounts/2,           'pour by volume',[0],  0,    False),
            Action([6], amounts/2,           'pour by volume',[0],  0,    False),
            Action([7], amounts/2,           'pour by volume',[0],  0,    False),
            Action([0,1,2], [[1e-3],[0.016]],'mix',           None, 0,    False),
            Action([0], [[0]],               'mix',           None, 0,    True)
        ]
        
        targets = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json").PRODUCTS

        super(SeparateTest_v0, self).__init__(
            shelf,
            actions,
            ["layers","targets"],
            targets,
            reward_function=e_rew,
            max_steps=5000
        )