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

Module to access and execute the ExtractWorld Engine

:title: extractworld_v1.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-06-24
"""

# pylint: disable=invalid-name
# pylint: disable=wrong-import-position

# import external modules
import os
import pickle
import sys
from random import choice
from copy import deepcopy

# import local modules
sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.chem_algorithms import material, util, vessel
from chemistrylab.reactions.reaction_base import _Reaction
from chemistrylab.general_bench.general_bench import *

from chemistrylab.reactions.get_reactions import convert_to_class,get_reactions
from chemistrylab.chem_algorithms.reward import ExtractionReward
import importlib


def wurtz_vessel(add_mat=""):
    """
    Function to generate an input vessel for the wurtz extraction experiment.
    Parameters
    ---------------
    None
    Returns
    ---------------
    `extract_vessel` : `vessel`
        A vessel object containing state variables, materials, solutes, and spectral data.
    Raises
    ---------------
    None
    """

    # initialize extraction vessel
    extraction_vessel = vessel.Vessel(label='extraction_vessel')

    # initialize DiEthylEther
    DiEthylEther = material.DiEthylEther()

    # initialize Na
    Na = material.Na()
    Na.set_charge(1.0)
    Na.set_solute_flag(True)
    Na.set_color(0.0)
    Na.set_polarity(2.0)
    Na.set_phase('l')
    Na._boiling_point = material.NaCl()._boiling_point

    # initialize Cl
    Cl = material.Cl()
    Cl.set_charge(-1.0)
    Cl.set_solute_flag(True)
    Cl.set_color(0.0)
    Cl.set_polarity(2.0)
    Cl.set_phase('l')
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
    add_material.set_phase('l')

    # material_dict
    material_dict = {
        DiEthylEther.get_name(): [DiEthylEther, 4.0, 'mol'],
        Na.get_name(): [Na, 1.0, 'mol'],
        Cl.get_name(): [Cl, 1.0, 'mol'],
        add_material.get_name(): [add_material, 1.0, 'mol']
    }

    # solute_dict
    solute_dict = {
        Na.get_name(): {DiEthylEther.get_name(): [DiEthylEther, 1.0, 'mol']},
        Cl.get_name(): {DiEthylEther.get_name(): [DiEthylEther, 1.0, 'mol']},
        add_material.get_name(): {DiEthylEther.get_name(): [DiEthylEther, 1.0, 'mol']}
    }

    material_dict, solute_dict, _ = util.check_overflow(
        material_dict=material_dict,
        solute_dict=solute_dict,
        v_max=extraction_vessel.get_max_volume()
    )

    # set events and push them to the queue
    extraction_vessel.push_event_to_queue(
        events=None,
        feedback=[
            ['update material dict', material_dict],
            ['update solute dict', solute_dict]
        ],
        dt=0
    )
    extraction_vessel.push_event_to_queue(
        events=None,
        feedback=None,
        dt=-100000
    )

    return extraction_vessel, add_mat


def make_solvent(mat):
    "Makes a Vessel with a single material"
    solvent_vessel = vessel.Vessel(
        label=f'{mat} Vessel',
        v_max=1e6,
        n_pixels=100,
        settling_switch=False,
        layer_switch=False,
    )
    # create the material dictionary for the solvent vessel
    solvent_class = convert_to_class(materials=[mat])[0]()
    solvent_material_dict = {mat:[solvent_class, 1e6]}

    # instruct the vessel to update its material dictionary
    event = Event('update material dict', solvent_material_dict,None)

    solvent_vessel.push_event_to_queue(feedback=[event], dt=0)
    return solvent_vessel


class GeneralWurtzExtract_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        e_rew= lambda x,y:ExtractionReward(vessels=x,desired_material=y,initial_target_amount=0).calc_reward()
        vessel_generators = [
            lambda x:wurtz_vessel(x)[0],
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Beaker 2"),
            lambda x:make_solvent("C6H14"),
            lambda x:make_solvent("diethyl ether")
        ]
        amounts=np.linspace(0.2,1,5).reshape([5,1])
        actions = [
            Action([0], amounts*10,          'drain by pixel',[1],  False),
            Action([0],-amounts,             'mix',           None, False),
            Action([1], amounts,             'pour by volume',[0],  False),
            Action([2], amounts,             'pour by volume',[0],  False),
            Action([0], amounts,             'pour by volume',[2],  False),
            Action([3], amounts/2,           'pour by volume',[0],  False),
            Action([4], amounts/2,           'pour by volume',[0],  False),
            Action([0], 32**amounts/200-0.01,'mix',           None, False),
            Action([0], [[0.01]],            'mix',           None, True)
        ]
        
        super(GeneralWurtzExtract_v2, self).__init__(
            vessel_generators,
            actions,
            importlib.import_module("chemistrylab.reactions.available_reactions.chloro_wurtz"),
            ["layers","targets"],
            n_visible=3,
            reward_function=e_rew
        )

