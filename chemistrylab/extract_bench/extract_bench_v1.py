'''
Module to access and execute the ExtractWorld Engine

:title: extractworld_v1.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-06-24
'''

# import extrernal modules
import numpy as np
import gym
import gym.spaces
import os
import pickle
import sys

sys.path.append("../../") # to access `chemistrylab`
from chemistrylab.chem_algorithms import material, util, vessel
from chemistrylab.extract_bench.extract_bench_v1_engine import ExtractBenchEnv

def get_extract_vessel(vessel_path=None, extract_vessel=None):
    '''
    Function to obtain an extraction vessel containing materials.

    Parameters
    ---------------
    `vessel_path` : `str` (default=`None`)
        A string indicating the local of a pickle file containing the required vessel object.
    `extract_vessel` : `vessel` (default=`None`)
        A vessel object containing state variables, materials, solutes, and spectral data.

    Returns
    ---------------
    `extract_vessel` : `vessel`
        A vessel object containing state variables, materials, solutes, and spectral data.

    Raises
    ---------------
    `IOError`:
        Raised if neither a path to a pickle file nor a vessel object is provided.
    '''

    # ensure that at least one of the methods to obtain a vessel is provided
    if all([not vessel_path, not vessel]):
        raise IOError("No vessel acquisition method specified.")

    # if a vessel object is provided, use it;
    # otherwise locate and extract the vessel object as a pickle file
    if not extract_vessel:
        with open(vessel_path, 'rb') as open_file:
            extract_vessel = pickle.load(open_file)

    return extract_vessel

def oil_vessel():
    '''
    Function to generate an input vessel for the oil and water extraction experiment.

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
    '''

    # initialize extraction vessel
    extraction_vessel = vessel.Vessel(label='extraction_vessel')

    # initialize materials
    H2O = material.H2O()
    Na = material.Na()
    Cl = material.Cl()
    Na.set_charge(1.0)
    Na.set_solute_flag(True)
    Na.set_polarity(2.0)
    Cl.set_charge(-1.0)
    Cl.set_solute_flag(True)
    Cl.set_polarity(2.0)

    # material_dict
    material_dict = {
        H2O.get_name(): [H2O, 30.0],
        Na.get_name(): [Na, 1.0],
        Cl.get_name(): [Cl, 1.0]
    }
    solute_dict = {
        Na.get_name(): [H2O.get_name(), 1.0],
        Cl.get_name(): [H2O.get_name(), 1.0],
    }

    material_dict, solute_dict, _ = util.check_overflow(
        material_dict=material_dict,
        solute_dict=solute_dict,
        v_max=extraction_vessel.get_max_volume()
    )

    event_1 = ['update material dict', material_dict]
    event_2 = ['update solute dict', solute_dict]
    event_3 = ['fully mix']

    extraction_vessel.push_event_to_queue(
        events=None,
        feedback=[event_1, event_2, event_3],
        dt=0
    )

    return extraction_vessel

class ExtractWorld_v1(ExtractBenchEnv):
    '''
    Class to define an environment which performs an extraction on materials in a vessel.
    '''

    def __init__(self):
        super(ExtractWorld_v1, self).__init__(
            extraction='wurtz',
            extraction_vessel=get_extract_vessel(
                vessel_path=os.path.join(os.getcwd(), "vessel_experiment_0.pickle"),
                extract_vessel=None
            ),
            solute="ethoxyethane",
            target_material='dodecane',
        )

class ExtractWorld_v2(ExtractBenchEnv):
    '''
    Class to define an environment which performs an extraction on materials in a vessel.
    '''

    def __init__(self):
        super(ExtractWorld_v2, self).__init__(
            extraction='water_oil',
            extraction_vessel=oil_vessel(),
            target_material='Na'
        )
